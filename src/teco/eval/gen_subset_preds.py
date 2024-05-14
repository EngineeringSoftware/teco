from typing import List, Optional

import seutil as su
from jsonargparse import CLI
from tqdm import tqdm

from teco.eval.specs import ModelsSpec
from teco.macros import Macros
from teco.utils import aggregate_metrics, summarize_metrics

logger = su.log.get_logger(__name__)


class SubsetPredictionsGenerator:
    """
    Generate the predictions on a subset of the evaluation set from the full set or a larger set.
    """
    def __init__(self, setup: str, from_set: str, to_set: str):
        self.setup = setup
        self.from_set = from_set
        self.to_set = to_set

        # analyze the eval locations
        self.from_eval_locs = su.io.load(
            Macros.work_dir / "setup" / setup / from_set / "eval_locs.jsonl"
        )
        self.to_eval_locs = su.io.load(
            Macros.work_dir / "setup" / setup / to_set / "eval_locs.jsonl"
        )
        self.subset_ids = [self.from_eval_locs.index(loc) for loc in self.to_eval_locs]

    def run_model(self, model: str, decoding: str):
        # load source predictions (no need to deserialize)
        from_preds = su.io.load(
            Macros.work_dir
            / "exp"
            / self.setup
            / self.from_set
            / model
            / decoding
            / "preds.jsonl"
        )
        assert len(from_preds) == len(
            self.from_eval_locs
        ), f"expected {len(self.from_eval_locs)} predictions, got {len(from_preds)}"

        # select subset
        out_dir = Macros.work_dir / "exp" / self.setup / self.to_set / model / decoding
        to_preds = [from_preds[idx] for idx in self.subset_ids]
        su.io.dump(out_dir / "preds.jsonl", to_preds)

        # recompute summary metrics
        metrics = aggregate_metrics([pred["metrics"] for pred in to_preds])
        metrics_summary = summarize_metrics(metrics)

        su.io.dump(
            out_dir / "metrics_summary.json", metrics_summary, su.io.Fmt.jsonNoSort
        )

    def run_models(
        self,
        models_spec_file: su.arg.RPath,
        decodings: List[str],
        exists: str = "skip",
        missing: str = "warning",
        skip_assert: bool = True,
        skip_rerank: bool = True,
        tags: Optional[List[str]] = None,
    ):
        assert exists in ["skip", "overwrite"]
        assert missing in ["warning", "ignore"]

        if tags is not None and "assert" in tags and skip_assert:
            raise ValueError("cannot skip_assert=True and specify tags=[assert,...]")
        if tags is not None and "rerank" in tags and skip_rerank:
            raise ValueError("cannot skip_rerank=True and specify tags=[rerank,...]")

        models_spec = su.io.load(models_spec_file, clz=ModelsSpec)
        to_dvc_add = []
        pbar = tqdm(total=len(models_spec))
        for model, model_spec in models_spec.items():
            if len(model_spec.trials) == 0:
                logger.warning(f"{model}: todo")
                pbar.update(1)
                continue

            if "assert" in model_spec.tags and skip_assert:
                logger.info(f"{model}: skipping assert models")
                pbar.update(1)
                continue

            if "rerank" in model_spec.tags and skip_rerank:
                logger.info(f"{model}: skipping rerank models")
                pbar.update(1)
                continue

            if tags is not None and len(set(model_spec.tags) & set(tags)) == 0:
                logger.info(f"{model}: not selected")
                pbar.update(1)
                continue

            for trial in model_spec.trials:
                for decoding in decodings:
                    pbar.set_description(f"{model} {trial} {decoding}")
                    from_dir = (
                        Macros.work_dir
                        / "exp"
                        / self.setup
                        / self.from_set
                        / trial
                        / decoding
                    )
                    if not from_dir.exists():
                        if missing == "warning":
                            logger.warning(f"Missing: {from_dir}")
                        elif missing == "ignore":
                            pass

                        pbar.update(1 / len(model_spec.trials) / len(decodings))
                        continue

                    to_dir = (
                        Macros.work_dir
                        / "exp"
                        / self.setup
                        / self.to_set
                        / trial
                        / decoding
                    )
                    if to_dir.exists():
                        if exists == "overwrite":
                            su.io.rmdir(to_dir)
                        elif exists == "skip":
                            logger.info(f"Skipped: {to_dir}")
                            pbar.update(1 / len(model_spec.trials) / len(decodings))
                            to_dvc_add.append(str(to_dir))
                            continue

                    self.run_model(trial, decoding)
                    pbar.update(1 / len(model_spec.trials) / len(decodings))
                    to_dvc_add.append(str(to_dir))
        pbar.close()

        print(f"# dvc command")
        print(f"  dvc add {' '.join(to_dvc_add)}")


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(SubsetPredictionsGenerator, as_positional=False)
