import collections
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import seutil as su
import torch.utils.data
import transformers
from jsonargparse.typing import Path_dc, Path_drw
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
    instantiate_class,
)
from teco.data.data import Data
from teco.data.utils import load_dataset
from teco.eval.metrics import batch_accuracy, batch_exact_match
from teco.macros import Macros
from teco.model.generation import GenerationMixin
from teco.model.processing import (
    Input,
    InputSequence,
    Output,
    get_input_sequence,
    get_output_ids,
    postprocess_recover_break_down_desc,
    reverse_insns_toks,
    subtoks2toks,
)
from teco.model.subtokenizer_bpe import SubtokenizerBPE
from teco.model.utils import (
    DefaultLightningCLI,
    LoadingAndProcessingOnTheFlyDataset,
    PathSafeSaveConfigCallback,
    ProcessingOnTheFlyDataset,
    shape_sequences,
)
from teco.utils import summarize_metrics
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BeamSearchScorer,
    GPT2LMHeadModel,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    NoBadWordsLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
)

logger = su.log.get_logger(__name__)


class CodeGPTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: Optional[Path_drw] = None,
        val_data_dir: Optional[Path_drw] = None,
        batch_size: int = 2,
        val_batch_size: int = 8,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        if isinstance(train_data_dir, Path_drw):
            train_data_dir = os.path.relpath(Path(train_data_dir.abs_path), Path.cwd())
        if isinstance(val_data_dir, Path_drw):
            val_data_dir = os.path.relpath(Path(val_data_dir.abs_path), Path.cwd())
        self.save_hyperparameters()

        self.tokenizer: SubtokenizerBPE = None
        self.train_dataset = None

    def prepare_data(self):
        self.tokenizer = self.trainer.lightning_module.tokenizer
        self.hparams.inputs = self.trainer.lightning_module.hparams.inputs
        self.hparams.output = self.trainer.lightning_module.hparams.output

        if self.hparams.train_data_dir is None:
            logger.warning("Train set not loaded")
            self.train_dataset = None
        else:
            with tqdm("Loading train set") as pbar:
                self.train_dataset = LoadingAndProcessingOnTheFlyDataset(
                    Path(self.hparams.train_data_dir),
                    clz=Data,
                    process_fn=self.process_train,
                    pbar=pbar,
                )

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = self.trainer.lightning_module.tokenizer
        self.hparams.inputs = self.trainer.lightning_module.hparams.inputs
        self.hparams.output = self.trainer.lightning_module.hparams.output

        if stage == "fit" or stage is None:
            self.train_dataset = self.trainer.strategy.broadcast(self.train_dataset, 0)
            print(
                f"{self.trainer.lightning_module.global_rank=}, {self.train_dataset=}"
            )

            if self.hparams.val_data_dir is None:
                logger.warning("Val set not loaded")
                self.val_dataset = None
            else:
                with tqdm("Loading val set") as pbar:
                    val_ds = load_dataset(
                        Path(self.hparams.val_data_dir),
                        clz=Data,
                        fully_deserialize=False,
                        pbar=pbar,
                    )
                # always val on each stmt of each data
                exp_val_data = []
                exp_val_stmt_i = []
                for d in val_ds:
                    for i in range(len(d.test_stmts)):
                        exp_val_data.append(d)
                        exp_val_stmt_i.append(i)

                self.val_dataset = ProcessingOnTheFlyDataset(
                    self.process, data=exp_val_data, stmt_i=exp_val_stmt_i
                )

    def teardown(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            self.train_dataset.close_files()

    def process_train(self, data: Data) -> Sequence[torch.Tensor]:
        # pick a random stmt to train each time
        stmt_i = random.randrange(0, len(data.test_stmts))
        return self.process(data, stmt_i)

    def process(self, data: Data, stmt_i: int) -> Sequence[torch.Tensor]:
        pad = self.tokenizer.pad_token_id

        # deserialize data on the fly (upon first seeing each data)
        data.finish_deserialization()

        # get inputs and outputs
        seq = get_input_sequence(data, stmt_i, self.hparams.inputs, self.tokenizer)
        src_tids = seq.get_stids()[:-1]  # skip the last eos

        assert self.hparams.output == Output.stmt
        tgt_tids = get_output_ids(data, stmt_i, self.hparams.output, self.tokenizer)
        tgt_tids = tgt_tids[1:]  # skip the first bos

        tids = src_tids + [self.tokenizer.sep_token_id] + tgt_tids
        tids = tids[-self.hparams.max_seq_len :]
        length = len(tids)
        tids = shape_sequences(tids, pad=pad, shapes=[self.hparams.max_seq_len])

        return (
            torch.tensor(tids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # persistent_workers=True,
            # pin_memory=True,  # Not working on multi gpu
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.hparams.val_batch_size,
            num_workers=0,
            # persistent_workers=True,
            # pin_memory=True,  # Not working on multi gpu
        )


class CodeGPTModule(GenerationMixin, pl.LightningModule):
    def __init__(
        self,
        pretrained_tokenizer: Union[Path_drw, str],
        pretrained_model: Union[Path_drw, str],
        optimizer_init: dict,
        lr_scheduler_init: dict,
        inputs: List[Input] = [Input.focalm, Input.sign, Input.prev_stmts],
        output: Output = Output.stmt,
    ):
        super().__init__()
        if isinstance(pretrained_tokenizer, Path_drw):
            pretrained_tokenizer = os.path.relpath(
                Path(pretrained_tokenizer.abs_path), Path.cwd()
            )
        if isinstance(pretrained_model, Path_drw):
            pretrained_model = os.path.relpath(
                Path(pretrained_model.abs_path), Path.cwd()
            )
        self.save_hyperparameters()

        self.tokenizer: SubtokenizerBPE = SubtokenizerBPE(
            AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer)
        )
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
            self.hparams.pretrained_model
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def configure_optimizers(self):
        if "weight_decay" in self.hparams.optimizer_init["init_args"]:
            no_decay = ["bias", "layer_norm.weight"]
            parameters = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.optimizer_init["init_args"][
                        "weight_decay"
                    ],
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            parameters = self.parameters()
        optimizer = instantiate_class(parameters, self.hparams.optimizer_init)
        lr_scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler_init)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int = -1,
        data_loader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        tids, _ = batch
        pad = self.tokenizer.pad_token_id
        labels = torch.where(tids == pad, -100, tids)
        outputs = self.model(tids, labels=labels, return_dict=True)
        loss = outputs.loss
        self.log_dict({"train/loss": loss.item()}, on_step=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int = -1,
        data_loader_idx: Optional[int] = None,
    ):
        tids, length = batch
        batch_size = tids.shape[0]
        pad = self.tokenizer.pad_token_id
        labels = torch.where(tids == pad, -100, tids)
        outputs = self.model(tids, labels=labels, return_dict=True)
        loss = outputs.loss

        # Metrics
        metrics_list = self.compute_metrics_list(
            tids,
            length,
            outputs.logits,
            log_text=(batch_idx <= 0) or (random.random() < 0.005),
        )
        metrics_list = {f"val/{k}": v for k, v in metrics_list.items()}
        metrics_list["val/loss"] = [loss.item()] * batch_size

        return metrics_list

    def compute_metrics_list(
        self,
        tids: torch.Tensor,
        length: torch.Tensor,
        logits: torch.Tensor,
        log_text: bool = True,
    ) -> Dict:
        """
        Compute metrics for a batch of inputs and logits.

        :param inputs: [batch_size * seq_len]
        :param logits: [batch_size * seq_len * vocab_size]
        :param lens: [batch_size]
        """
        _, preds_tids = torch.topk(logits, k=1, dim=2)
        preds_tids = preds_tids.squeeze(dim=2)  # [batch_size * seq_len]
        preds = []
        golds = []
        batch_size = tids.shape[0]

        for i in range(batch_size):
            preds.append(self.tokenizer.id2subtok(preds_tids[i][: length[i] - 1]))
            golds.append(self.tokenizer.id2subtok(tids[i][1 : length[i]]))

        metrics_list = {}
        # subtoken-level accuracy
        metrics_list["acc"] = batch_accuracy(golds, preds)
        metrics_list["xmatch"] = batch_exact_match(golds, preds)

        if log_text:
            s = ""
            for i in range(batch_size):
                s += f"# Example {i}\n\n"
                s += f"- gold\n```\n{''.join(golds[i])}\n```\n\n"
                s += f"- pred\n```\n{''.join(preds[i])}\n```\n\n"
                s += f"- metrics\n\n"
                for k, v in metrics_list.items():
                    s += f"{k}: {v[i]}\n"
                s += "\n"

            self.logger.experiment.add_text(
                "examples/val", s, global_step=self.global_step
            )

        return metrics_list

    def validation_epoch_end(self, outputs):
        metrics_list = collections.defaultdict(list)
        for o in outputs:
            for k in o:
                metrics_list[k] += o[k]

        metrics = summarize_metrics(metrics_list)
        self.log_dict(metrics)

    def save_pretrained(self, save_dir: Union[str, Path, Path_drw, Path_dc]):
        if isinstance(save_dir, (Path_drw, Path_dc)):
            save_dir = Path(save_dir.abs_path)
        self.model.save_pretrained(save_dir)
        self.tokenizer.tokenizer.save_pretrained(save_dir)

    # ========== generation ==========

    def gen_postprocess(self, gen_out: dict) -> None:
        subtoks = self.tokenizer.id2subtok(gen_out["tids"][1:-1])
        toks = subtoks2toks(subtoks)
        if self.hparams.output in {Output.insn, Output.revinsn}:
            toks = postprocess_recover_break_down_desc(toks)
        if self.hparams.output == Output.revinsn:
            toks = reverse_insns_toks(toks)
        gen_out["toks"] = toks

    def gen_get_past(self, seq: InputSequence, max_length: int = 200) -> dict:
        # run LM to encode input sequence
        max_seq_len = self.model.config.n_positions - max_length

        src_tids = seq.get_stids()[:-1]  # remove EOS
        src_tids = src_tids[-max_seq_len:]
        src_tids = torch.tensor([src_tids], dtype=torch.long, device=self.device)
        out = self.model(src_tids, return_dict=True, use_cache=True)
        return {"past": out.past_key_values}

    @classmethod
    def expand_past(
        cls, past: Tuple[Tuple[torch.Tensor]], expand_size: int
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(p.expand(expand_size, -1, -1, -1) for p in layer_past)
            for layer_past in past
        )

    def gen_preprocess(self, decode_params: dict) -> dict:
        decode_params["model_kwargs"] = self.gen_get_past(
            decode_params.pop("seq"), decode_params.get("max_length", 200)
        )
        return decode_params

    def generate_greedy(
        self, model_kwargs: dict, max_length: int = 200, **kwargs
    ) -> List[dict]:
        self.generation_warn_unused_kwargs(kwargs)

        # pass to transformers' generation util
        out = self.model.greedy_search(
            torch.tensor(
                [[self.tokenizer.sep_token_id]], dtype=torch.long, device=self.device
            ),
            logits_processor=LogitsProcessorList(
                [
                    NoBadWordsLogitsProcessor(
                        [[self.tokenizer.pad_token_id]], self.tokenizer.eos_token_id
                    )
                ]
            ),
            stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length)]),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **model_kwargs,
        )

        def compute_seq_score(out: GreedySearchDecoderOnlyOutput) -> float:
            score = torch.tensor(0.0, dtype=torch.float, device=self.device)
            for j in range(1, out.sequences.shape[1]):
                score += F.log_softmax(out.scores[j - 1][0], dim=0)[out.sequences[0, j]]
            return score.item()

        return [
            {
                "tids": [
                    x
                    for x in out.sequences[0].tolist()
                    if x != self.tokenizer.pad_token_id
                ],
                "score": compute_seq_score(out),
            }
        ]

    def generate_sampling(
        self,
        model_kwargs: dict,
        max_length: int = 200,
        num_return_sequences: int = 10,
        batch_size: int = 20,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[dict]:
        self.generation_warn_unused_kwargs(kwargs)

        generated = {}

        logits_warpers = []
        if top_p is not None:
            logits_warpers.append(TopPLogitsWarper(top_p))
        if top_k is not None:
            logits_warpers.append(TopKLogitsWarper(top_k))
        if temperature is not None:
            logits_warpers.append(TemperatureLogitsWarper(temperature))
        logits_warper = LogitsProcessorList(logits_warpers)

        def compute_seq_score(out: SampleDecoderOnlyOutput, i: int) -> float:
            score = torch.tensor(0.0, dtype=torch.float, device=self.device)
            for j in range(1, out.sequences.shape[1]):
                score += F.log_softmax(out.scores[j - 1][i], dim=0)[out.sequences[i, j]]
                if out.sequences[i, j] == self.tokenizer.eos_token_id:
                    break
            return score.item()

        input_ids = torch.tensor(
            [[self.tokenizer.sep_token_id]], dtype=torch.long, device=self.device
        )
        for start_i in range(0, num_return_sequences, batch_size):
            this_batch_size = min(batch_size, num_return_sequences - start_i)

            logits_processors = []
            logits_processors.append(
                NoBadWordsLogitsProcessor(
                    [[self.tokenizer.pad_token_id]], self.tokenizer.eos_token_id
                )
            )
            logits_processor = LogitsProcessorList(logits_processors)

            # pass to transformers' generation util
            exp_input_ids, exp_model_kwargs = self.model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=this_batch_size,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )
            # past need to be manually expanded
            exp_model_kwargs["past"] = self.expand_past(
                exp_model_kwargs["past"], this_batch_size
            )

            out = self.model.sample(
                exp_input_ids,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length)]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **exp_model_kwargs,
            )

            for i in range(this_batch_size):
                seq = out.sequences[i].tolist()

                # avoid producing duplicate samples
                key = tuple(seq)
                if key in generated:
                    continue
                if key in generated:
                    generated[key]["weight"] += 1
                else:
                    generated[key] = {
                        "tids": [x for x in seq if x != self.tokenizer.pad_token_id],
                        "score": compute_seq_score(out, i),
                        "weight": 1,
                    }

        return list(generated.values())

    def generate_beam_search(
        self,
        model_kwargs: dict,
        max_length: int = 200,
        num_return_sequences: int = 10,
        beam_size: int = 10,
        num_beam_groups: int = 1,
        repetition_penalty: float = 1.0,
        diversity_penalty: float = 0.0,
        **kwargs,
    ) -> List[dict]:
        self.generation_warn_unused_kwargs(kwargs)

        # pass to transformers' generation util
        input_ids = torch.tensor(
            [[self.tokenizer.sep_token_id]], dtype=torch.long, device=self.device
        )
        exp_input_ids, exp_model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=beam_size,
            is_encoder_decoder=self.model.config.is_encoder_decoder,
            **model_kwargs,
        )
        # past need to be manually expanded
        exp_model_kwargs["past"] = self.expand_past(exp_model_kwargs["past"], beam_size)

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=beam_size,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences,
            device=self.device,
        )
        logits_processors = []
        logits_processors.append(
            NoBadWordsLogitsProcessor(
                [[self.tokenizer.pad_token_id]], self.tokenizer.eos_token_id
            )
        )
        if diversity_penalty != 0.0:
            logits_processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty,
                    num_beams=beam_size,
                    num_beam_groups=num_beam_groups,
                )
            )
        if repetition_penalty != 1.0:
            logits_processors.append(
                RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        logits_processor = LogitsProcessorList(logits_processors)

        amp_ctx = None
        if num_beam_groups == 1:
            beam_search_func = self.model.beam_search
        else:
            if torch.cuda.is_available():
                amp_ctx = torch.cuda.amp.autocast(enabled=False)
                amp_ctx.__enter__()
            beam_search_func = self.model.group_beam_search

        out = beam_search_func(
            exp_input_ids,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length)]),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **exp_model_kwargs,
        )

        if amp_ctx is not None:
            amp_ctx.__exit__(None, None, None)

        return [
            {
                "tids": [
                    x
                    for x in out.sequences[i].tolist()
                    if x != self.tokenizer.pad_token_id
                ],
                "score": out.sequences_scores[i].item(),
                "weight": 1,
            }
            for i in range(out.sequences.shape[0])
        ]


if __name__ == "__main__":
    su.log.setup(Macros.log_file)

    OPTIMIZER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.Optimizer, override=True
    )
    LR_SCHEDULER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
    )

    DefaultLightningCLI(
        CodeGPTModule,
        CodeGPTDataModule,
        save_config_callback=PathSafeSaveConfigCallback,
        optimizers=[(None, "optimizer", "model.optimizer_init")],
        lr_schedulers=[(None, "lr_scheduler", "model.lr_scheduler_init")],
    )
