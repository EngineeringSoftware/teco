import collections
import dataclasses
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pytorch_lightning as pl
import seutil as su
import torch
from jsonargparse.typing import Path_drw
from pytorch_lightning.callbacks import BasePredictionWriter
from teco.data.data import Data
from teco.data.utils import load_dataset
from teco.eval.metrics import bleu, code_bleu, edit_sim, rouge_l
from teco.model.utils import ProcessingOnTheFlyDataset
from teco.utils import aggregate_metrics, summarize_metrics
from tqdm import tqdm

logger = su.log.get_logger(__name__)


@dataclasses.dataclass
class PredictInputs:
    idx: int = -1
    data: Data = None
    gold_stmt: List[str] = None
    gold_insn: List[str] = None


class PredictionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        predict_data_dir: Optional[Path_drw] = None,
        train_data_dir: Optional[Path_drw] = None,
        batch_size: int = 1,  # unused
        val_batch_size: int = 1,  # unused
    ):
        super().__init__()
        if isinstance(predict_data_dir, Path_drw):
            predict_data_dir = os.path.relpath(
                Path(predict_data_dir.abs_path), Path.cwd()
            )
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage == "predict" or stage is None:
            data_dir = Path(self.hparams.predict_data_dir)
            with tqdm("Loading predict set") as pbar:
                dataset = load_dataset(data_dir, clz=Data, pbar=pbar)
                gold_stmts = su.io.load(data_dir / "gold_stmts.jsonl")
                gold_insns = su.io.load(data_dir / "gold_insns.jsonl")

            self.predict_dataset = ProcessingOnTheFlyDataset(
                PredictInputs,
                idx=range(len(dataset)),
                data=dataset,
                gold_stmt=gold_stmts,
                gold_insn=gold_insns,
            )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=None,  # No batching/auto-collate during evaluation
            num_workers=1,  # Disabling multiprocessing to get valid batch_indices
            persistent_workers=True,
            # pin_memory=True,  # Not working on multi gpu
        )


@dataclasses.dataclass
class Prediction:
    """Prediction for one data"""

    id: int = -1
    data_id: str = -1
    input_stids: List[int] = dataclasses.field(default_factory=list)
    topk: List[dict] = dataclasses.field(default_factory=list)
    topk_insn: List[dict] = dataclasses.field(default_factory=list)
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    time: float = -1  # time for prediction in seconds
    misc: dict = dataclasses.field(default_factory=dict)  # whatever other info

    @classmethod
    def load_predictions_metrics_only(cls, pred_file: Path) -> List["Prediction"]:
        preds_unserialized = su.io.load(pred_file, serialization=False)
        preds = []
        for data in preds_unserialized:
            preds.append(
                cls(
                    id=data["id"],
                    data_id=data["data_id"],
                    metrics=data["metrics"],
                    time=data["time"],
                    misc=data["misc"],
                )
            )
        return preds


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Union[Path, str], no_compute_metrics: bool = False):
        super().__init__(write_interval="epoch")
        self.no_compute_metrics = no_compute_metrics
        self.output_dir = Path(output_dir)
        su.io.mkdir(self.output_dir)
        self.temp_dir = self.output_dir / "temp"
        su.io.mkdir(self.temp_dir)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        results: List[List[List[Prediction]]],
        batch_indices: Optional[Sequence[Sequence[Sequence[int]]]],
    ):
        # collect preds, and put into a file according to current global rank
        preds: List[Prediction] = []
        for dl_batch_preds in results:
            for batch_preds in dl_batch_preds:
                if isinstance(batch_preds, list):
                    for pred in batch_preds:
                        preds.append(pred)
                else:
                    preds.append(batch_preds)

        su.io.dump(
            self.temp_dir / f"{pl_module.global_rank}.pkl",
            preds,
        )

        # wait all processes to finish prediction
        trainer.strategy.barrier("prediction")

        if pl_module.global_rank == 0:
            id2pred = {}
            for rank in range(trainer.world_size):
                for pred in su.io.load(self.temp_dir / f"{rank}.pkl"):
                    id2pred[pred.id] = pred
            if sorted(id2pred.keys()) != list(range(len(id2pred))):
                logger.warning(f"Prediction ids are not continuous")
            preds = [id2pred[i] for i in sorted(id2pred.keys())]

            # dump predictions
            logger.info("Saving predictions")
            su.io.dump(
                self.output_dir / "preds.jsonl",
                preds,
            )

            # compute summary metrics
            metrics = aggregate_metrics([pred.metrics for pred in preds])
            metrics_summary = summarize_metrics(metrics)

            su.io.dump(
                self.output_dir / "metrics_summary.json",
                metrics_summary,
                su.io.Fmt.jsonNoSort,
            )

            # delete temp directory
            su.io.rmdir(self.temp_dir)


def compute_similarity_metrics_all(
    gold: List[str], topk: List[List[str]]
) -> Dict[str, List[float]]:
    # compute all required metrics
    if len(topk) == 0:
        # dummy empty prediction, to simplify corner case handling
        topk = [[]]
    metrics_all = collections.defaultdict(list)
    for pred in topk:
        metrics_all["xmatch"].append(100 if gold == pred else 0)
        metrics_all["bleu"].append(bleu(gold, pred))
        metrics_all["code-bleu"].append(code_bleu(gold, pred))
        metrics_all["edit-sim"].append(edit_sim(gold, pred))
        for k, v in rouge_l(gold, pred).items():
            metrics_all[f"rouge-{k}"].append(v)

    return metrics_all


def compute_similarity_metrics(
    gold: List[str],
    topk: List[List[str]],
    k_values: List[int] = None,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    if k_values is None:
        k_values = [3, 5, 10, 100]

    metrics = {}
    metrics_all = compute_similarity_metrics_all(gold, topk)

    # xmatch and xmatch-topk
    metrics["xmatch"] = metrics_all["xmatch"][0]
    for k in k_values:
        metrics[f"xmatch-top{k}"] = max(metrics_all["xmatch"][:k])

    # similarity metrics
    for m in ["bleu", "code-bleu", "edit-sim", "rouge-p", "rouge-r", "rouge-f"]:
        # top1
        metrics[m] = metrics_all[m][0]
        # max, min, avg
        metrics[f"{m}-max"] = max(metrics_all[m])
        metrics[f"{m}-min"] = min(metrics_all[m])
        metrics[f"{m}-avg"] = sum(metrics_all[m]) / len(metrics_all[m])
        # weighted avg
        if weights is not None:
            metrics[f"{m}-wavg"] = sum(
                [x * w for x, w in zip(metrics_all[m], weights)]
            ) / sum(weights)

    return metrics
