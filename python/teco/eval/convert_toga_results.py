import pandas as pd
import seutil as su
from jsonargparse import CLI
from tqdm import tqdm

from teco.data.data import Data
from teco.data.utils import load_dataset
from teco.macros import Macros
from teco.model.prediction import Prediction, compute_similarity_metrics
from teco.utils import aggregate_metrics, summarize_metrics


class TOGAResultsConverter:
    def convert(
        self,
        setup: str,
        model: str,
        eval_set: str = "eval-assert-stmt/test",
        decoding: str = "bs10-last",
    ):
        data_dir = Macros.work_dir / "setup" / setup / eval_set
        with tqdm(desc="load dataset") as pbar:
            dataset = load_dataset(data_dir, clz=Data, only=["id"], pbar=pbar)
            gold_stmts = su.io.load(data_dir / "gold_stmts.jsonl")

        model_dir = Macros.work_dir / "exp" / setup / "train" / model
        df = pd.read_csv(model_dir / "results" / "assertion_preds.csv")
        df_best = df.loc[df.groupby("idxs").logit_1.idxmax()].set_index("idxs")

        preds = []
        for i in tqdm(range(len(dataset)), desc="converting"):
            data = dataset[i]
            gold = gold_stmts[i]
            try:
                seq = df_best.loc[i].pred_assertion.split() + [";"]
            except KeyError:
                seq = []

            metrics = compute_similarity_metrics(gold, [seq])
            pred = Prediction(
                id=i, data_id=data.id, topk=[{"toks": seq}], metrics=metrics
            )
            preds.append(pred)

        output_dir = Macros.work_dir / "exp" / setup / eval_set / model / decoding

        # dump predictions
        su.io.dump(output_dir / "preds.jsonl", preds)

        # compute summary metrics
        metrics = aggregate_metrics([pred.metrics for pred in preds])
        metrics_summary = summarize_metrics(metrics)

        su.io.dump(
            output_dir / "metrics_summary.json", metrics_summary, su.io.Fmt.jsonNoSort
        )


if __name__ == "__main__":
    CLI(TOGAResultsConverter, as_positional=False)
