import collections
import copy
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw
from pytorch_lightning.utilities import AttributeDict
from tqdm import tqdm

from teco.data.data import Data
from teco.data.structures import Consts
from teco.data.utils import load_dataset, save_dataset
from teco.macros import Macros

logger = su.log.get_logger(__name__)


class EvalSetup:
    def __init__(self, mask_str_literal: bool = False):
        self.config = AttributeDict({k: v for k, v in locals().items() if k != "self"})

    def prepare_train(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
    ):
        """
        Prepares the train and validation sets under the requested setup.
        :param data_dir: source data directory.
        :param split_dir: split ids directory.
        :param out_dir: output data directory, which will contain train/ and val/ datasets.
        """
        train_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(train_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load ids
        sn2ids: Dict[str, List[str]] = {}
        for sn in [Macros.train, Macros.val]:
            sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")
        all_ids: Set[str] = set(sum(sn2ids.values(), []))

        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[Data] = load_dataset(
                data_dir, clz=Data, expected_ids=all_ids, pbar=pbar
            )
            if self.config.mask_str_literal:
                pbar.reset(len(dataset))
                pbar.set_description("Masking string literals")
                for d in dataset:
                    self.mask_str_literal_in_data(d)
                    pbar.update(1)
        indexed_dataset: Dict[int, Data] = {d.id: d for d in dataset}

        sn2ds = {sn: [indexed_dataset[i] for i in ids] for sn, ids in sn2ids.items()}

        # Save data
        for sn in [Macros.train, Macros.val]:
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, sn2ds[sn])

    def prepare_eval(
        self,
        data_dir: Path_drw,
        split_dir: Path_drw,
        out_dir: Path_dc,
        first_assertion: bool = False,
        runnable: bool = False,
        min_stmt: int = 0,
        max_eval_data: int = 100_000,
        max_per_proj: int = -1,
        seed: int = time.time_ns(),
    ):
        """
        Prepares the evaluation (val & test) sets under the requested setup.

        Each method-level data is break down to a couple of line-level data, where X-th line is the line to be predicted and the previous X-1 lines and some initial tokens in the X-th line are the context.

        If max_eval_data or max_per_proj forced less line-level data being extracted than the max possible number, the down-sampling process extractes similar number of lines from each method.

        :param data_dir: source data directory.
        :param split_dir: split ids directory.
        :param out_dir: output data directory, which will contain val/ and test/ datasets.
        :param first_assertion: if True, will only use the first assertion in each test as the line to predict.
        :param runnable: if True, will only use data whose runtime data is collected and valid.
        :param min_stmt: minimum number of statements as context.
        :param max_eval_data: maximum number of data in val/test set.
        :param max_per_proj: maximum number of data from one project in val/test set.
        :param seed: random seed (for sampling data).
        """
        assert min_stmt >= 0
        assert max_eval_data > 0
        assert max_per_proj > 0 or max_per_proj == -1

        eval_config = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "data_dir", "split_dir", "out_dir"]
        }
        self.config.update(eval_config)
        logger.info(f"Config:\n{self.config}")

        data_dir = Path(data_dir.abs_path)
        split_dir = Path(split_dir.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)
        su.io.dump(out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # Load data ids
        sn2ids: Dict[str, List[int]] = {}
        for sn in [Macros.val, Macros.test]:
            sn2ids[sn] = su.io.load(split_dir / f"{sn}.json")
        all_ids: Set[int] = set(sum(sn2ids.values(), []))

        # Load data
        with tqdm("Loading data") as pbar:
            dataset: List[Data] = load_dataset(
                data_dir, clz=Data, expected_ids=all_ids, pbar=pbar
            )
            if self.config.mask_str_literal:
                pbar.reset(len(dataset))
                pbar.set_description("Masking string literals")
                for d in dataset:
                    self.mask_str_literal_in_data(d)
                    pbar.update(1)
        indexed_dataset: Dict[int, Data] = {d.id: d for d in dataset}

        sn2ds = {sn: [indexed_dataset[i] for i in ids] for sn, ids in sn2ids.items()}

        for sn in [Macros.val, Macros.test]:
            # Reset random state
            random.seed(seed)

            # Find all available eval data_id: (data_id, stmt_i) tuples
            id2locs: Dict[int, List[Tuple[str, int]]] = collections.defaultdict(list)

            for data in sn2ds[sn]:
                if self.config.runnable:
                    if not data.runtime_data_valid():
                        continue

                found_assertion = False
                for stmt_i, stmt in enumerate(data.test_stmt_toks):
                    if self.config.first_assertion:
                        if not found_assertion:
                            if not self.is_assertion(stmt):
                                continue
                            else:
                                found_assertion = True
                        else:
                            break

                    if stmt_i < self.config.min_stmt:
                        continue

                    id2locs[data.id].append((data.id, stmt_i))

            # Sample eval locations from each data id in turn
            eval_locs: List[Tuple[str, int]] = []
            id2proj: Dict[int, str] = {d.id: d.proj_name for d in sn2ds[sn]}
            proj2cnt: Dict[str, int] = collections.defaultdict(int)
            while True:
                should_finish = True
                for id, locs in id2locs.items():
                    if (
                        self.config.max_per_proj != -1
                        and proj2cnt[id2proj[id]] >= self.config.max_per_proj
                    ):
                        continue
                    if len(locs) > 0:
                        should_finish = False

                        if len(locs) == 0:
                            i = 0
                        else:
                            i = random.randint(0, len(locs) - 1)
                        eval_locs.append(locs[i])
                        proj2cnt[id2proj[id]] += 1
                        del locs[i]
                    if len(eval_locs) >= self.config.max_eval_data:
                        should_finish = True
                        break
                if should_finish:
                    break

            # Assemble the final eval set of data and gold
            eval_ds: List[Data] = []
            gold_stmts: List[List[str]] = []
            gold_insns: List[List[str]] = []
            gold_fqinsns: List[List[str]] = []
            eval_locs.sort()
            for data_id, stmt_i in eval_locs:
                data = copy.deepcopy(indexed_dataset[data_id])
                gold_stmt = data.test_stmt_toks[stmt_i]
                gold_insn = data.test_stmt_insns[stmt_i]
                gold_fqinsn = data.test_stmt_fqinsns[stmt_i]
                data.cutoff(stmt_i)

                eval_ds.append(data)
                gold_stmts.append(gold_stmt)
                gold_insns.append(gold_insn)
                gold_fqinsns.append(gold_fqinsn)

            # Save
            su.io.mkdir(out_dir / sn)
            save_dataset(out_dir / sn, eval_ds)
            su.io.dump(out_dir / sn / "gold_stmts.jsonl", gold_stmts)
            su.io.dump(out_dir / sn / "gold_insns.jsonl", gold_insns)
            su.io.dump(out_dir / sn / "gold_fqinsns.jsonl", gold_fqinsns)
            su.io.dump(out_dir / sn / "eval_locs.jsonl", eval_locs)

    @classmethod
    def is_assertion(cls, stmt: List[str]) -> bool:
        return (
            stmt[0] == "Assert"
            or stmt[0].startswith("assert")
            or stmt[0].startswith("fail")
        )

    @classmethod
    def mask_str_literal_in_data(cls, data: Data):
        for i, node in enumerate(data.focalm.traverse()):
            if (
                node.ast_type == Consts.ast_string_literal_expr
                and node.tok_kind == Consts.tok_literal
            ):
                node.tok = '"STR"'
        for i, node in enumerate(data.test_sign.traverse()):
            if (
                node.ast_type == Consts.ast_string_literal_expr
                and node.tok_kind == Consts.tok_literal
            ):
                node.tok = '"STR"'
        for stmt in data.test_stmts:
            for i, node in enumerate(stmt.traverse()):
                if (
                    node.ast_type == Consts.ast_string_literal_expr
                    and node.tok_kind == Consts.tok_literal
                ):
                    node.tok = '"STR"'
        for insns in data.test_stmt_insns:
            for i in range(2, len(insns)):
                if insns[i - 2] == "LDC" and insns[i - 1] == "String":
                    insns[i] = "STR"

        # extra data
        if data.setup_methods is not None:
            for m in data.setup_methods:
                for i, node in enumerate(m.traverse()):
                    if (
                        node.ast_type == Consts.ast_string_literal_expr
                        and node.tok_kind == Consts.tok_literal
                    ):
                        node.tok = '"STR"'
        if data.teardown_methods is not None:
            for m in data.teardown_methods:
                for i, node in enumerate(m.traverse()):
                    if (
                        node.ast_type == Consts.ast_string_literal_expr
                        and node.tok_kind == Consts.tok_literal
                    ):
                        node.tok = '"STR"'
        if data.last_called_methods is not None:
            for m in data.last_called_methods:
                if m is None:
                    continue
                for i, node in enumerate(m.traverse()):
                    if (
                        node.ast_type == Consts.ast_string_literal_expr
                        and node.tok_kind == Consts.tok_literal
                    ):
                        node.tok = '"STR"'
        if data.similar_stmts is not None:
            for x in data.similar_stmts:
                if x is None:
                    continue
                for i, node in enumerate(x[1].traverse()):
                    if (
                        node.ast_type == Consts.ast_string_literal_expr
                        and node.tok_kind == Consts.tok_literal
                    ):
                        node.tok = '"STR"'
        if data.runtime_types_values_delta is not None:
            for rv in data.runtime_types_values_delta:
                for name, (type_, value) in rv.items():
                    if value.startswith('"'):
                        rv[name] = (type_, '"STR"')


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(EvalSetup, as_positional=False)
