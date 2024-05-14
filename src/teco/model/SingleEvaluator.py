import os
import time
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import seutil as su
from jsonargparse.typing import Path_drw, Path_fr
from teco.macros import Macros
from teco.model.load_trained import load_trained
from teco.model.prediction import (
    PredictInputs,
    Prediction,
    PredictionDataModule,
    PredictionWriter,
    compute_similarity_metrics,
)
from teco.model.processing import Output, get_input_sequence
from teco.model.utils import DefaultLightningCLI, PathSafeSaveConfigCallback

logger = su.log.get_logger(__name__)


class SingleEvaluator(pl.LightningModule):
    """
    For evaluating a single 1-input 1-output model.
    Supported models: CodeT5, CodeGPT
    """

    def __init__(
        self,
        model_cls: str,
        model_ckpt: Union[Path_drw, Path_fr],
        decode_method: str = "greedy",
        decode_params: Optional[dict] = None,
    ):
        super().__init__()
        model_ckpt = os.path.relpath(Path(model_ckpt.abs_path), Path.cwd())
        if decode_params is None:
            decode_params = {}
        self.save_hyperparameters()

        self.model = load_trained(model_cls, model_ckpt)
        self.tokenizer = self.model.tokenizer
        self.hparams.inputs = self.model.hparams.inputs
        self.hparams.output = self.model.hparams.output

    def predict_step(
        self,
        pred_inputs: PredictInputs,
        batch_idx: int = -1,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        data = pred_inputs.data
        pred = Prediction(id=pred_inputs.idx, data_id=data.id)
        if self.hparams.output == Output.stmt:
            # put the gold insn into data, in case the model needs it (evaluating a insn->stmt model)
            pred_inputs.data.test_stmt_insns.append(pred_inputs.gold_insn)

        time_beg = time.time()

        # prepare input sequence
        seq = get_input_sequence(
            data,
            len(data.test_stmts),
            self.hparams.inputs,
            self.tokenizer,
        )
        pred.input_stids = seq.get_stids()

        # run model generation
        topk = self.model.generate(
            self.hparams.decode_method,
            seq=seq,
            **self.hparams.decode_params,
        )

        if self.hparams.output == Output.stmt:
            pred.topk = topk
        elif self.hparams.output == Output.insn:
            pred.topk_insn = topk
        else:
            raise ValueError(f"Unsupported output: {self.hparams.output}")

        time_end = time.time()
        pred.time = time_end - time_beg

        # compute metrics
        pred.metrics = {}
        if self.hparams.output == Output.stmt:
            pred.metrics.update(
                compute_similarity_metrics(
                    pred_inputs.gold_stmt, [x["toks"] for x in pred.topk]
                )
            )
            pred.metrics["num-seq"] = len(pred.topk)
        elif self.hparams.output == Output.insn:
            pred.metrics.update(
                {
                    f"insn-{k}": v
                    for k, v in compute_similarity_metrics(
                        pred_inputs.gold_insn, [x["toks"] for x in pred.topk_insn]
                    ).items()
                }
            )
            pred.metrics["insn-num-seq"] = len(pred.topk_insn)
        else:
            raise ValueError(f"Unsupported output: {self.hparams.output}")

        return pred


if __name__ == "__main__":
    su.log.setup(Macros.log_file)

    DefaultLightningCLI(
        SingleEvaluator,
        PredictionDataModule,
        save_config_callback=PathSafeSaveConfigCallback,
        prediction_writer=PredictionWriter,
    )
