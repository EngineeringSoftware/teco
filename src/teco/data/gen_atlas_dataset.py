from typing import List, Optional, Tuple

import seutil as su
from jsonargparse import CLI
from tqdm import tqdm

from teco.data.data import BASIC_FIELDS, Data
from teco.data.eval_setup import EvalSetup
from teco.data.utils import load_dataset
from teco.macros import Macros


class ATLASDatasetGenerator:
    def generate(
        self,
        setup: str,
        out_dir: su.arg.RPath,
        val_set: str = "eval-assert-stmt/val",
        test_set: str = "eval-assert-stmt/test",
    ):
        su.io.mkdir(out_dir)
        data_dir = Macros.work_dir / "setup" / setup

        # train set
        pbar = tqdm(desc="preparing train set")
        train_dataset: List[Data] = load_dataset(
            data_dir / "train" / "train", clz=Data, only=BASIC_FIELDS, pbar=pbar
        )
        train_src = []
        train_tgt = []
        pbar.reset(len(train_dataset))
        for data in train_dataset:
            # find first assertion, if any
            for stmt_i, stmt in enumerate(data.test_stmt_toks):
                if EvalSetup.is_assertion(stmt):
                    # add as a training data
                    src, tgt = self.format_data(data, stmt_i)
                    train_src.append(src)
                    train_tgt.append(tgt)
                    break

            pbar.update(1)
        pbar.close()
        su.io.dump(
            out_dir / "Training" / "testMethods.txt", train_src, su.io.Fmt.txtList
        )
        su.io.dump(
            out_dir / "Training" / "assertLines.txt", train_tgt, su.io.Fmt.txtList
        )
        print(f"# train data: {len(train_src)} (from {len(train_dataset)} methods)")

        # val set
        pbar = tqdm(desc="preparing val set")
        val_dataset: List[Data] = load_dataset(
            data_dir / val_set, clz=Data, only=BASIC_FIELDS, pbar=pbar
        )
        val_golds = su.io.load(data_dir / val_set / "gold_stmts.jsonl")
        val_src = []
        val_tgt = []
        for data, gold in zip(val_dataset, val_golds):
            src, tgt = self.format_data(data, len(data.test_stmts), gold)
            pbar.update(1)
            val_src.append(src)
            val_tgt.append(tgt)
        pbar.close()
        su.io.dump(out_dir / "Eval" / "testMethods.txt", val_src, su.io.Fmt.txtList)
        su.io.dump(out_dir / "Eval" / "assertLines.txt", val_tgt, su.io.Fmt.txtList)
        print(f"# val data: {len(val_src)}")

        # test set
        pbar = tqdm(desc="preparing test set")
        test_dataset: List[Data] = load_dataset(
            data_dir / test_set, clz=Data, only=BASIC_FIELDS, pbar=pbar
        )
        test_golds = su.io.load(data_dir / test_set / "gold_stmts.jsonl")
        test_src = []
        test_tgt = []
        for data, gold in zip(test_dataset, test_golds):
            src, tgt = self.format_data(data, len(data.test_stmts), gold)
            pbar.update(1)
            test_src.append(src)
            test_tgt.append(tgt)
        pbar.close()
        su.io.dump(out_dir / "Testing" / "testMethods.txt", test_src, su.io.Fmt.txtList)
        su.io.dump(out_dir / "Testing" / "assertLines.txt", test_tgt, su.io.Fmt.txtList)
        print(f"# test data: {len(test_src)}")

    def format_data(
        self, data: Data, stmt_i: int, gold: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        src_toks = (
            data.test_sign_toks
            + ["{"]
            + sum(data.test_stmt_toks[:stmt_i], [])
            + ["<AssertPlaceHolder>"]
            + ["}"]
            + data.focalm_toks
        )
        if gold is None:
            tgt_toks = data.test_stmt_toks[stmt_i]
        else:
            tgt_toks = gold
        return " ".join(src_toks), " ".join(tgt_toks)


if __name__ == "__main__":
    CLI(ATLASDatasetGenerator, as_positional=False)
