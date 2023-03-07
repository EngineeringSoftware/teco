import seutil as su
from jsonargparse import CLI
from tqdm import tqdm

from teco.model.prediction import Prediction, compute_similarity_metrics
from teco.utils import aggregate_metrics, summarize_metrics


class CodeGPTPredictionsFixer:
    def fix(self, data_dir: su.arg.RPath, pred_dir: su.arg.RPath):
        golds = su.io.load(data_dir / "gold_stmts.jsonl")
        preds = su.io.load(pred_dir / "preds.jsonl", clz=Prediction)

        for gold, pred in tqdm(zip(golds, preds), total=len(golds)):
            # cut off each sequence at the first ";"
            for seq in pred.topk:
                toks = seq["toks"]
                fixed_toks = []
                for tok in toks:
                    if tok.startswith(";"):
                        fixed_toks.append(";")
                        break
                    else:
                        fixed_toks.append(tok)
                seq["toks"] = fixed_toks

            # recompute metrics
            topk_toks = [seq["toks"] for seq in pred.topk]
            pred.metrics = compute_similarity_metrics(gold, topk_toks)

        su.io.dump(pred_dir / "preds.jsonl", preds)

        # recompute summary metrics
        metrics = aggregate_metrics([pred.metrics for pred in preds])
        metrics_summary = summarize_metrics(metrics)

        su.io.dump(
            pred_dir / "metrics_summary.json",
            metrics_summary,
            su.io.Fmt.jsonNoSort,
        )


if __name__ == "__main__":
    CLI(CodeGPTPredictionsFixer, as_positional=False)
