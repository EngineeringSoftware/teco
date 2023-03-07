import collections
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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
from teco.eval.metrics import (
    batch_accuracy,
    batch_bleu,
    batch_edit_sim,
    batch_exact_match,
    batch_rouge_l,
    batch_sentence_code_bleu,
)
from teco.macros import Macros
from teco.model.generation import GenerationMixin
from teco.model.loss_scaler import LossScaler
from teco.model.processing import (
    Input,
    InputSequence,
    Output,
    get_input_sequence,
    get_output_ids,
    postprocess_recover_break_down_desc,
    preprocess_break_down_desc,
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
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BeamSearchScorer,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    NoBadWordsLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
    T5ForConditionalGeneration,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation import (
    GreedySearchEncoderDecoderOutput,
    SampleEncoderDecoderOutput,
)

logger = su.log.get_logger(__name__)


class CodeT5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: Optional[Path_drw] = None,
        val_data_dir: Optional[Path_drw] = None,
        batch_size: int = 1,
        val_batch_size: int = 8,
        max_seq_len: int = 512,
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

        # prepare inputs
        seq = get_input_sequence(data, stmt_i, self.hparams.inputs, self.tokenizer)

        # get inputs (sub)token ids
        src_tids = seq.get_stids()
        src_tids = src_tids[-self.hparams.max_seq_len :]
        src_len = len(src_tids)
        src_tids = shape_sequences(src_tids, pad=pad, shapes=[self.hparams.max_seq_len])

        # get outputs (sub)token ids
        tgt_tids = get_output_ids(data, stmt_i, self.hparams.output, self.tokenizer)
        tgt_tids = tgt_tids[: self.hparams.max_seq_len]
        tgt_len = len(tgt_tids)
        tgt_tids = shape_sequences(tgt_tids, pad=pad, shapes=[self.hparams.max_seq_len])

        return (
            torch.tensor(src_tids, dtype=torch.long),
            torch.tensor(src_len, dtype=torch.long),
            torch.tensor(tgt_tids, dtype=torch.long),
            torch.tensor(tgt_len, dtype=torch.long),
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


class CodeT5Module(GenerationMixin, pl.LightningModule):
    def __init__(
        self,
        pretrained_tokenizer: Union[Path_drw, str],
        pretrained_model: Union[Path_drw, str],
        optimizer_init: dict,
        lr_scheduler_init: dict,
        inputs: List[Input] = [Input.focalm, Input.sign, Input.prev_stmts],
        output: Output = Output.stmt,
        loss_scale_step: int = -1,  # recommended 5000
        loss_scale_important: float = 2.0,
        loss_scale_unimportant: float = 0.2,
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
        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(self.hparams.pretrained_model)
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.hparams.loss_scale_step != -1:
            assert self.hparams.output == Output.insn, "loss scale only support insn"
            self.loss_scaler = LossScaler(
                self.tokenizer,
                self.hparams.loss_scale_important,
                self.hparams.loss_scale_unimportant,
            )

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
        src_tids, _, tgt_tids, tgt_len = batch
        pad = self.tokenizer.pad_token_id
        outputs = self.model(
            input_ids=src_tids,
            attention_mask=src_tids.ne(pad).float(),
            decoder_input_ids=tgt_tids,
            return_dict=True,
        )
        loss = self.compute_loss(
            logits=outputs.logits, tgt_tids=tgt_tids, labels_len=tgt_len
        )

        self.log_dict({"loss/train": loss.item()}, on_step=True, sync_dist=True)
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
        src_tids, src_len, tgt_tids, tgt_len = batch
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        batch_size = src_tids.shape[0]

        attention_mask = src_tids.ne(pad).float()
        enc_out = self.model.encoder(
            src_tids, attention_mask=attention_mask, return_dict=True
        )

        # run model in teacher-forcing mode (same as training, to get loss)
        tf_out = self.model(
            encoder_outputs=enc_out,
            attention_mask=attention_mask,
            decoder_input_ids=tgt_tids,
            return_dict=True,
        )
        loss = self.compute_loss(
            logits=tf_out.logits, tgt_tids=tgt_tids, labels_len=tgt_len
        )

        # run greedy decoding to get prediction
        greedy_out = self.model.greedy_search(
            torch.full((batch_size, 1), bos, dtype=torch.long, device=self.device),
            logits_processor=LogitsProcessorList(
                [NoBadWordsLogitsProcessor([[pad]], eos)]
            ),
            stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(200)]),
            pad_token_id=pad,
            eos_token_id=eos,
            return_dict_in_generate=True,
            output_scores=True,
            encoder_outputs=enc_out,
            attention_mask=attention_mask,
        )

        # compute similarity metrics
        # log the inputs&outputs for 0.5% batches and always include the first batch
        log_text = (batch_idx <= 0) or (random.random() < 0.005)
        metrics_list, report = self.compute_metrics_list(
            tgt_tids,
            tgt_len,
            src_tids,
            src_len,
            pred_tids=greedy_out.sequences[:, 1:],
            logits=tf_out.logits,
            log_text=log_text,
        )
        metrics_list = {f"val/{k}": v for k, v in metrics_list.items()}
        metrics_list["loss/val"] = [loss.item()] * batch_size
        if log_text:
            report = report.replace("## Example", f"## Batch {batch_idx} Example")

        return metrics_list, report

    def compute_loss(
        self,
        logits: torch.Tensor,  # [B, S]
        labels: Optional[torch.Tensor] = None,  # [B, S]
        labels_len: Optional[torch.Tensor] = None,  # [B]
        tgt_tids: Optional[torch.Tensor] = None,  # [B, S]
    ) -> torch.Tensor:
        assert (labels is None) != (tgt_tids is None)

        if labels is None:
            pad = self.tokenizer.pad_token_id
            labels = tgt_tids.roll(-1, dims=1)
            labels[:, -1] = pad
            labels = torch.where(labels == pad, -100, labels)
        if labels_len is None:
            labels_len = torch.ones_like(labels[:, 0]) * labels.shape[-1]
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.reshape_as(labels)
        if (
            self.hparams.loss_scale_step != -1
            and self.global_step >= self.hparams.loss_scale_step
        ):
            loss = self.loss_scaler(labels, loss)

        return loss.sum() / labels_len.sum()

    def compute_metrics_list(
        self,
        tgt_tids: torch.Tensor,
        tgt_len: torch.Tensor,
        src_tids: torch.Tensor,
        src_len: torch.Tensor,
        pred_tids: torch.Tensor = None,
        logits: torch.Tensor = None,
        log_text: bool = True,
    ) -> Tuple[dict, str]:
        """
        Compute metrics during validation.
        """
        # for computing teacher-forcing accuracy
        _, tf_tids = torch.topk(logits, k=1, dim=2)
        tf_tids = tf_tids.squeeze(dim=2)  # [batch_size * seq_len]

        preds = []
        golds = []
        srcs = []
        tfs = []
        batch_size = tgt_tids.shape[0]

        for i in range(batch_size):
            gold_sts = self.tokenizer.id2subtok(tgt_tids[i][1 : tgt_len[i] - 1])
            golds.append(postprocess_recover_break_down_desc(subtoks2toks(gold_sts)))
            src_sts = self.tokenizer.id2subtok(src_tids[i][1 : src_len[i] - 1])
            srcs.append(postprocess_recover_break_down_desc(subtoks2toks(src_sts)))
            tf_sts = self.tokenizer.id2subtok(tf_tids[i][: tgt_len[i] - 2])
            tfs.append(postprocess_recover_break_down_desc(subtoks2toks(tf_sts)))

            pred_sts = []
            for subtok in self.tokenizer.id2subtok(pred_tids[i]):
                if subtok in {self.tokenizer.eos_token, self.tokenizer.pad_token}:
                    break
                pred_sts.append(subtok)
            preds.append(postprocess_recover_break_down_desc(subtoks2toks(pred_sts)))

        metrics_list = {}
        # accuracy metrics between gold and teacher-forcing
        metrics_list["tf-acc"] = batch_accuracy(golds, tfs)
        metrics_list["tf-xmatch"] = batch_exact_match(golds, tfs)

        # similarity metrics between gold and pred
        metrics_list["xmatch"] = batch_exact_match(golds, preds)
        metrics_list["bleu"] = batch_bleu(golds, preds)
        metrics_list["code-bleu"] = batch_sentence_code_bleu(golds, preds)
        metrics_list["edit-sim"] = batch_edit_sim(golds, preds)
        for k, v in batch_rouge_l(golds, preds).items():
            metrics_list[f"rouge-{k}"] = v

        report = ""
        if log_text:
            for i in range(batch_size):
                report += f"## Example {i}\n\n"
                report += f"- src\n```\n{' '.join(srcs[i])}\n```\n\n"
                report += f"- gold\n```\n{' '.join(golds[i])}\n```\n\n"
                report += f"- pred\n```\n{' '.join(preds[i])}\n```\n\n"
                report += f"- metrics\n\n"
                for k, v in metrics_list.items():
                    report += f"{k}: {v[i]}\n"
                report += "\n"

        return metrics_list, report

    def validation_epoch_end(self, outputs):
        metrics_list = collections.defaultdict(list)
        report = ""
        for metrics_list_batch, report_batch in outputs:
            for k in metrics_list_batch:
                metrics_list[k] += metrics_list_batch[k]
            report += report_batch

        metrics = summarize_metrics(metrics_list)
        self.log_dict(metrics, sync_dist=True)

        self.logger.experiment.add_text(
            "examples/val", report, global_step=self.global_step
        )

    def save_pretrained(self, save_dir: Union[str, Path, Path_drw, Path_dc]):
        if isinstance(save_dir, (Path_drw, Path_dc)):
            save_dir = Path(save_dir.abs_path)
        self.model.save_pretrained(save_dir)
        self.tokenizer.tokenizer.save_pretrained(save_dir)

    # ========== generation ==========

    def gen_run_encoder(self, seq: InputSequence) -> dict:
        # run encoder
        pad = self.tokenizer.pad_token_id
        max_seq_len = self.model.config.n_positions
        src_tids = seq.get_stids()[-max_seq_len:]
        src_tids = shape_sequences(src_tids, pad=pad, shapes=[max_seq_len])
        src_tids = torch.tensor([src_tids], dtype=torch.long, device=self.device)
        attention_mask = src_tids.ne(pad).float()
        enc_out = self.model.get_encoder()(
            src_tids, attention_mask=attention_mask, return_dict=True
        )
        return {"attention_mask": attention_mask, "encoder_outputs": enc_out}

    def gen_preprocess(self, decode_params: dict) -> dict:
        decode_params["model_kwargs"] = self.gen_run_encoder(decode_params.pop("seq"))
        return decode_params

    def gen_postprocess(self, gen_out: dict) -> None:
        subtoks = self.tokenizer.id2subtok(gen_out["tids"][1:-1])
        toks = subtoks2toks(subtoks)
        if self.hparams.output in {Output.insn, Output.revinsn}:
            toks = postprocess_recover_break_down_desc(toks)
        if self.hparams.output == Output.revinsn:
            toks = reverse_insns_toks(toks)
        gen_out["toks"] = toks

    def generate_greedy(
        self, model_kwargs: dict, max_length: int = 200, **kwargs
    ) -> List[dict]:
        self.generation_warn_unused_kwargs(kwargs)

        # pass to transformers' generation util
        out = self.model.greedy_search(
            torch.tensor(
                [[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device
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

        def compute_seq_score(out: GreedySearchEncoderDecoderOutput) -> float:
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
                "weight": 1,
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

        def compute_seq_score(out: SampleEncoderDecoderOutput, i: int) -> float:
            score = torch.tensor(0.0, dtype=torch.float, device=self.device)
            for j in range(1, out.sequences.shape[1]):
                score += F.log_softmax(out.scores[j - 1][i], dim=0)[out.sequences[i, j]]
                if out.sequences[i, j] == self.tokenizer.eos_token_id:
                    break
            return score.item()

        input_ids = torch.tensor(
            [[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device
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
            [[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device
        )
        exp_input_ids, exp_model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=beam_size,
            is_encoder_decoder=self.model.config.is_encoder_decoder,
            **model_kwargs,
        )
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

    def score(
        self,
        seq: InputSequence,
        tgts: List[List[str]],
        batch_size: int = 20,
    ) -> torch.Tensor:
        assert self.hparams.output in {Output.stmt, Output.insn}
        pad = self.tokenizer.pad_token_id
        scores = torch.zeros(len(tgts), dtype=torch.float, device=self.device)

        def _get_output_ids(toks: List[str]) -> List[int]:
            toks = preprocess_break_down_desc(toks)
            return (
                [self.tokenizer.bos_token_id]
                + self.tokenizer.toks2stids(toks)
                + [self.tokenizer.eos_token_id]
            )

        model_kwargs = self.gen_run_encoder(seq)
        max_seq_len = self.model.config.n_positions

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        dummy_input_ids = torch.tensor([[0]], dtype=torch.long, device=self.device)
        for start_i in range(0, len(tgts), batch_size):
            this_batch_size = min(batch_size, len(tgts) - start_i)
            _, exp_model_kwargs = self.model._expand_inputs_for_generation(
                dummy_input_ids,
                expand_size=this_batch_size,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )
            exp_input_ids = torch.full(
                (this_batch_size, max_seq_len),
                pad,
                dtype=torch.long,
                device=self.device,
            )
            tgt_len = torch.zeros(this_batch_size, dtype=torch.long, device=self.device)
            for i, tgt in enumerate(tgts[start_i : start_i + this_batch_size]):
                tgt_tids = _get_output_ids(tgt)[:max_seq_len]
                tgt_len[i] = len(tgt_tids)
                exp_input_ids[i, : len(tgt_tids)] = torch.tensor(
                    tgt_tids, dtype=torch.long, device=self.device
                )

            decoder_outputs = self.model(
                **self.model.prepare_inputs_for_generation(
                    exp_input_ids, **exp_model_kwargs
                ),
                return_dict=True,
            )

            # compute generation score using the loss function
            logits = decoder_outputs.logits
            labels = exp_input_ids.roll(-1, dims=1)
            labels[:, -1] = pad
            labels = torch.where(labels == pad, -100, labels)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss.reshape_as(labels)
            loss = -torch.div(loss.sum(dim=1), tgt_len)
            scores[start_i : start_i + this_batch_size] = loss
        return scores


if __name__ == "__main__":
    su.log.setup(Macros.log_file)

    OPTIMIZER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.Optimizer, override=True
    )
    LR_SCHEDULER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
    )

    DefaultLightningCLI(
        CodeT5Module,
        CodeT5DataModule,
        save_config_callback=PathSafeSaveConfigCallback,
        optimizers=[(None, "optimizer", "model.optimizer_init")],
        lr_schedulers=[(None, "lr_scheduler", "model.lr_scheduler_init")],
    )
