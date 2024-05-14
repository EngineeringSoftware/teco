from typing import List

import seutil as su
from jsonargparse import CLI
from tqdm import tqdm

from teco.macros import Macros
from teco.model.prediction import Prediction, compute_similarity_metrics
from teco.utils import aggregate_metrics, summarize_metrics

logger = su.log.get_logger(__name__)


class RunnableReranker:
    def rerank(
        self,
        src_model: str,
        tgt_model: str,
        runnable_eval_set: str = "eval-runnable-any-stmt/test",
        other_eval_sets: List[str] = [
            "eval-any-stmt/test",
            "eval-assert-stmt/test",
            "eval-runnable-assert-stmt/test",
        ],
        setup: str = "CSNm",
        decoding: str = "bs10-last",
    ):
        # make sure dataset and predictions on all eval sets are available
        runnable_pred_dir = (
            Macros.work_dir / "exp" / setup / runnable_eval_set / src_model / decoding
        )
        other_pred_dirs = [
            Macros.work_dir / "exp" / setup / eval_set / src_model / decoding
            for eval_set in other_eval_sets
        ]
        for pred_dir in [runnable_pred_dir] + other_pred_dirs:
            if not pred_dir.exists():
                raise FileNotFoundError(f"Predictions not found: {pred_dir}")

        runnable_data_dir = Macros.work_dir / "setup" / setup / runnable_eval_set
        other_data_dirs = [
            Macros.work_dir / "setup" / setup / eval_set for eval_set in other_eval_sets
        ]
        for data_dir in [runnable_data_dir] + other_data_dirs:
            if not data_dir.exists():
                raise FileNotFoundError(f"Dataset not found: {data_dir}")

        # load runnable preds, and eval_locs
        runnable_preds = su.io.load(runnable_pred_dir / "preds.jsonl", clz=Prediction)
        runnable_eval_locs = su.io.load(runnable_data_dir / "eval_locs.jsonl")

        # rerank each eval_set's preds (don't forget runnable eval set itself at last)
        for eval_set in tqdm(other_eval_sets + [runnable_eval_set], desc="reranking"):
            src_pred_dir = (
                Macros.work_dir / "exp" / setup / eval_set / src_model / decoding
            )
            tgt_pred_dir = (
                Macros.work_dir / "exp" / setup / eval_set / tgt_model / decoding
            )
            data_dir = Macros.work_dir / "setup" / setup / eval_set

            eval_locs = su.io.load(data_dir / "eval_locs.jsonl")
            gold_stmts = su.io.load(data_dir / "gold_stmts.jsonl")
            src_preds = su.io.load(src_pred_dir / "preds.jsonl", clz=Prediction)
            tgt_preds = []

            for eval_loc, gold_stmt, src_pred in tqdm(
                zip(eval_locs, gold_stmts, src_preds),
                total=len(eval_locs),
                desc=f"reranking {eval_set}",
            ):
                try:
                    runnable_i = runnable_eval_locs.index(eval_loc)
                except ValueError:
                    tgt_preds.append(src_pred)
                    continue

                runnable_pred = runnable_preds[runnable_i]
                new_best = 0
                try:
                    new_best = runnable_pred.misc.get("topk_runnable", []).index(100)
                except ValueError:
                    try:
                        new_best = runnable_pred.misc.get("topk_compilable", []).index(
                            100
                        )
                    except ValueError:
                        pass

                if new_best == 0:
                    # no reranking happened
                    tgt_preds.append(src_pred)
                else:
                    # move the #new_best prediction to the top
                    tgt_pred = Prediction(
                        id=src_pred.id,
                        data_id=src_pred.data_id,
                        input_stids=src_pred.input_stids,
                        topk=[src_pred.topk[new_best]]
                        + src_pred.topk[:new_best]
                        + src_pred.topk[new_best + 1 :],
                        time=src_pred.time,
                        misc=src_pred.misc,
                    )
                    for m in ["topk_runnable", "topk_compilable", "topk_runtime"]:
                        if m in tgt_pred.misc:
                            tgt_pred.misc["topk_runnable"] = (
                                [tgt_pred.misc[m][new_best]]
                                + tgt_pred.misc[m][:new_best]
                                + tgt_pred.misc[m][new_best + 1 :]
                            )
                    # metrics need to be recomputed
                    tgt_pred.metrics = compute_similarity_metrics(
                        gold_stmt, [seq["toks"] for seq in tgt_pred.topk]
                    )
                    tgt_preds.append(tgt_pred)

            # save predictions
            su.io.dump(tgt_pred_dir / "preds.jsonl", tgt_preds)

            # recompute summary metrics
            metrics = aggregate_metrics([pred.metrics for pred in tgt_preds])
            metrics_summary = summarize_metrics(metrics)

            su.io.dump(
                tgt_pred_dir / "metrics_summary.json",
                metrics_summary,
                su.io.Fmt.jsonNoSort,
            )


if __name__ == "__main__":
    CLI(RunnableReranker, as_positional=False)
