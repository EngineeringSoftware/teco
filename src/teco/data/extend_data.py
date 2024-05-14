import collections
import contextlib
import copy
import dataclasses
import functools
import os
import random
import re
import traceback
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import seutil as su
from jsonargparse import CLI
from tqdm import tqdm
from transformers import AutoTokenizer

from teco.data.ast_visitor import Visitor
from teco.data.bm25 import BM25
from teco.data.data import BASIC_FIELDS, Data
from teco.data.raw_data_loader import (
    ClassResolutionException,
    FieldResolutionException,
    MethodResolutionException,
    RawDataLoader,
)
from teco.data.structures import AST, Consts, FieldStructure, Insn, Scope
from teco.data.utils import load_dataset, save_dataset
from teco.exe.bcconstrainer import BytecodeConstrainer
from teco.exe.constraints import InsnConstraint
from teco.macros import Macros
from teco.model.prediction import Prediction
from teco.model.subtokenizer_bpe import SubtokenizerBPE

logger = su.log.get_logger(__name__)


@dataclasses.dataclass
class ExtendedDataScope:
    name: str
    fields: List[str]
    deps: List[str]
    require_bcconstrainer: bool = False
    require_raw_data: bool = False


EXTENDED_DATA_GROUPS = {
    "types_local": ExtendedDataScope(
        name="types_local",
        fields=["types_local"],
        deps=BASIC_FIELDS,
        require_bcconstrainer=True,
        require_raw_data=False,
    ),
    "setup_teardown_methods": ExtendedDataScope(
        name="setup_teardown_methods",
        fields=[
            "setup_methods",
            "setup_mids",
            "teardown_methods",
            "teardown_mids",
        ],
        deps=BASIC_FIELDS,
        require_bcconstrainer=False,
        require_raw_data=True,
    ),
    "fields_set_notset": ExtendedDataScope(
        name="fields_set_notset",
        fields=["fields_set", "fields_notset"],
        deps=BASIC_FIELDS + ["setup_mids"],
        require_bcconstrainer=False,
        require_raw_data=True,
    ),
    "types_absent": ExtendedDataScope(
        name="types_absent",
        fields=["types_absent"],
        deps=BASIC_FIELDS + ["types_local"],
        require_bcconstrainer=False,
        require_raw_data=True,
    ),
    "last_called_methods": ExtendedDataScope(
        name="last_called_methods",
        fields=["last_called_methods"],
        deps=BASIC_FIELDS,
        require_bcconstrainer=False,
        require_raw_data=True,
    ),
    "similar_stmts": ExtendedDataScope(
        name="similar_stmts",
        fields=["similar_stmts"],
        deps=BASIC_FIELDS,
        require_bcconstrainer=False,
        require_raw_data=True,
    ),
}


RE_SUBTOKENIZER = re.compile(
    r"(?<=[_$])(?!$)|(?<!^)(?=[_$])|(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[0-9]|[A-Z][a-z0-9])|(?<=[0-9])(?=[a-zA-Z])|\b"
)


class ExtendedDataCollector:
    def __init__(self, data_dir: su.arg.RPath, raw_data_dir: su.arg.RPath):
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir

        self.rdloader = RawDataLoader(indexed=True)
        self.rdloader.load_jre_data(self.raw_data_dir / "jre")

        self.cur_log_file = None

    def log_to_file(self, msg: str):
        if self.cur_log_file is None:
            logger.warning(f"[no log file selected] {msg}")
        else:
            su.io.dump(
                self.cur_log_file,
                [msg],
                su.io.Fmt.txtList,
                append=True,
            )

    def collect_single(self, name: str):
        group = EXTENDED_DATA_GROUPS[name]

        self.cur_log_file = self.data_dir / "log" / f"extra-{name}.txt"
        su.io.rm(self.cur_log_file)

        dataset = load_dataset(
            self.data_dir, clz=Data, only=group.deps, exclude=group.fields
        )

        if group.require_bcconstrainer:
            bcconstrainer_func = lambda: BytecodeConstrainer()
        else:
            bcconstrainer_func = lambda: contextlib.nullcontext()

        if not group.require_raw_data:
            self.rdloader = None
            self.cg = None

        invalid_count = 0
        with bcconstrainer_func() as bcconstrainer:
            for d in tqdm(dataset):
                if group.require_raw_data:
                    if self.rdloader.load_project_data(self.raw_data_dir / d.proj_name):
                        if name == "similar_stmts":
                            self.collect_similar_stmts_per_project()
                try:
                    if name == "types_local":
                        self.collect_types_local(d, bcconstrainer)
                    elif name == "fields_set_notset":
                        self.collect_fields_set_notset(d, 2)
                    elif name == "setup_teardown_methods":
                        self.collect_setup_teardown_methods(d)
                    elif name == "types_absent":
                        self.collect_types_absent(d)
                    elif name == "last_called_methods":
                        self.collect_last_called_methods(d, max_tok=200)
                    elif name == "similar_stmts":
                        self.collect_similar_stmts(d)
                    else:
                        logger.error(f"Unknown item to collect: {name}")
                        break
                except KeyboardInterrupt:
                    input(
                        "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                    )
                    self.log_to_file(f"data {d.id} invalid due to: User interrupted")
                    invalid_count += 1
                except Exception as e:
                    self.log_to_file(
                        f"data {d.id} invalid due to: {e}; {traceback.format_exc()}"
                    )
                    invalid_count += 1

        if invalid_count > 0:
            self.log_to_file("=====")
            self.log_to_file(f"total invalid: {invalid_count}")
            logger.warning(f"total invalid: {invalid_count}")

        save_dataset(self.data_dir, dataset, only=group.fields)

    def extract_locals_types(self, insn_c: InsnConstraint) -> List[str]:
        state_local = []
        holes = []
        for local in insn_c.locals:
            if local.is_uninitialized():
                holes.append("uninitialized")
                continue
            state_local += holes
            holes = []
            state_local += [local.get_name()]
        return state_local

    def collect_types_local(self, data: Data, bcconstrainer: BytecodeConstrainer):
        test_cname = Insn.class_i2qname(data.test_mkey.split("#")[0])
        test_desc = "(" + data.test_mkey.split("(")[-1]

        data.types_local = []
        sid = bcconstrainer.start_session([test_cname, test_desc, "0"])

        try:
            insn_c = bcconstrainer.try_step(sid, [])
            data.types_local.append(self.extract_locals_types(insn_c))

            for insn in data.test_stmt_fqinsns:
                insn_c = bcconstrainer.try_step(sid, insn)
                data.types_local.append(self.extract_locals_types(insn_c))
                bcconstrainer.submit_step(sid)
        except RuntimeError:
            self.log_to_file(
                f"Failed to collect types_local for {data.test_mkey}: {traceback.format_exc()}"
            )
            data.types_local = [[] for _ in range(len(data.test_stmts) + 1)]
        finally:
            bcconstrainer.end_session(sid)

    def collect_fields_set_notset(self, data: Data, field_search_level: int = 4):
        """must be after collect_setup_teardown_methods"""
        # find fields of interest (in the test and focal class and their ancestors)
        fids_all: Set[int] = set()
        fid2fs: Dict[int, FieldStructure] = {}

        for cid in [data.test_cid, data.focal_cid]:
            while cid >= 0:
                cs = self.rdloader.get_class(cid)
                for fid in cs.fields:
                    fs = self.rdloader.get_field(fid)
                    if fs.is_static():
                        continue
                    fids_all.add(fid)
                    fid2fs[fid] = fs

                cid = cs.ext

        fids_notset: Set[int] = copy.copy(fids_all)
        data.fields_set = []
        data.fields_notset = []

        # analyze setup method and find field usages
        if data.setup_mids is not None:
            for mid in data.setup_mids:
                ms = self.rdloader.get_method(mid)
                if ms.bytecode is not None:
                    self.find_putfield_rec(
                        insns=list(ms.bytecode.insns.values()),
                        fids_notset=fids_notset,
                        level_max=field_search_level,
                    )

        data.fields_notset.append(sorted([fid2fs[fid].name for fid in fids_notset]))
        fids_set = fids_all - fids_notset
        data.fields_set.append(sorted([fid2fs[fid].name for fid in fids_set]))

        # analyze each statement and find field usages
        for i in range(len(data.test_stmts)):
            # analyze the bytecode
            self.find_putfield_rec(
                insns=Insn.convert_from_tokens(data.test_stmt_fqinsns[i]),
                fids_notset=fids_notset,
                level_max=field_search_level,
            )

            data.fields_notset.append(sorted([fid2fs[fid].name for fid in fids_notset]))
            fids_set = fids_all - fids_notset
            data.fields_set.append(sorted([fid2fs[fid].name for fid in fids_set]))

    def find_putfield_rec(
        self,
        insns: List[Insn],
        fids_notset: Set[int],
        level_max: int = 4,
        level: int = 0,
        visited_mids: Set[int] = None,
    ) -> None:
        if visited_mids is None:
            visited_mids = set()

        for insn in insns:
            if insn.op == Consts.op_putfield:
                try:
                    cid = self.rdloader.lookup_class(insn.get_owner())
                except ClassResolutionException:
                    continue
                try:
                    fid = self.rdloader.lookup_field(cid, insn.get_field_name())
                except FieldResolutionException:
                    continue
                if fid in fids_notset:
                    fids_notset.remove(fid)
            elif insn.op in Consts.ops_method_insn:
                if level < level_max:
                    # try to find the bytecode of the invoked method
                    try:
                        cid = self.rdloader.lookup_class(insn.get_owner())
                    except ClassResolutionException:
                        continue
                    try:
                        mid = self.rdloader.lookup_method(
                            cid, insn.get_method_namedesc(), insn.op
                        )
                    except MethodResolutionException:
                        self.log_to_file(
                            f"Failed to analyze method invocation insn {insn}"
                        )
                        continue
                    if mid < 0:
                        continue
                    if mid in visited_mids:
                        continue

                    ms = self.rdloader.get_method(mid)
                    if ms.bytecode is not None:
                        self.find_putfield_rec(
                            insns=list(ms.bytecode.insns.values()),
                            fids_notset=fids_notset,
                            level_max=level_max,
                            level=level + 1,
                            visited_mids=visited_mids,
                        )
                    visited_mids.add(mid)

    def collect_setup_teardown_methods(self, data: Data):
        setup_methods = []
        setup_mids = []
        teardown_methods = []
        teardown_mids = []

        # include the setup/teardown methods from ancestors
        testc_and_ancestors = []
        cur_cid = data.test_cid
        while cur_cid >= 0:
            cur_cs = self.rdloader.get_class(cur_cid)
            testc_and_ancestors.append(cur_cs)
            cur_cid = cur_cs.ext

        for cs in testc_and_ancestors:
            for mid in cs.methods:
                ms = self.rdloader.get_method(mid)

                # look for class-level setup methods: marked with @BeforeClass / @BeforeAll
                if (
                    "org.junit.BeforeClass" in ms.atypes
                    or "org.junit.jupiter.api.BeforeAll" in ms.atypes
                ):
                    setup_methods.append(ms)
                    setup_mids.append(mid)

        for cs in testc_and_ancestors:
            for mid in cs.methods:
                ms = self.rdloader.get_method(mid)
                # look for setup methods: marked with @Before / @BeforeEach
                if (
                    "org.junit.Before" in ms.atypes
                    or "org.junit.jupiter.api.BeforeEach" in ms.atypes
                ):
                    setup_methods.append(ms)
                    setup_mids.append(mid)

        for cs in testc_and_ancestors:
            for mid in cs.methods:
                ms = self.rdloader.get_method(mid)
                # look for teardown methods: marked with @After / @AfterEach
                if (
                    "org.junit.After" in ms.atypes
                    or "org.junit.jupiter.api.AfterEach" in ms.atypes
                ):
                    teardown_methods.append(ms)
                    teardown_mids.append(mid)

        for cs in testc_and_ancestors:
            for mid in cs.methods:
                ms = self.rdloader.get_method(mid)
                # look for class-level teardown methods: marked with @AfterClass / @AfterAll
                if (
                    "org.junit.AfterClass" in ms.atypes
                    or "org.junit.jupiter.api.AfterAll" in ms.atypes
                ):
                    teardown_methods.append(ms)
                    teardown_mids.append(mid)

        data.setup_methods = []
        for ms in setup_methods:
            if ms.ast is not None:
                data.setup_methods.append(ms.ast)
            else:
                self.log_to_file(
                    f"setup method {ms.namedesc} ({data.proj_name=}, {data.id=}) has no source code"
                )
        data.setup_mids = setup_mids

        data.teardown_methods = []
        for ms in teardown_methods:
            if ms.ast is not None:
                data.teardown_methods.append(ms.ast)
            else:
                self.log_to_file(
                    f"teardown method {ms.namedesc} ({data.proj_name=}, {data.id=}) has no source code"
                )
        data.teardown_mids = teardown_mids

    def collect_types_absent(self, data: Data):
        # TODO: do we need to consider type hierarchy? If so, should we prepare the type hierarchy beforehand or as-needed?
        data.types_absent = []

        # collect the types needed by the focal method at the beginning
        types_absent = set()
        focalm = self.rdloader.get_method(data.focal_mid)
        if not focalm.is_static():
            focalc = self.rdloader.get_class(data.focal_cid)
            types_absent.add(focalc.name)

        for ptype in focalm.ptypes:
            types_absent.add(ptype)

        # remove the types that are already used by the test class fields
        testc = self.rdloader.get_class(data.test_cid)
        for fid in testc.fields:
            fs = self.rdloader.get_field(fid)
            if fs.type in types_absent:
                types_absent.remove(fs.type)

        data.types_absent.append(list(sorted(copy.copy(types_absent))))

        for stmt_i in range(len(data.test_stmts)):
            # remove the types that are already created and saved as local variables
            types_local = data.types_local[stmt_i]
            for t in types_local:
                if t in types_absent:
                    types_absent.remove(t)

            data.types_absent.append(list(sorted(copy.copy(types_absent))))

            # if focalm is invoked in this statement, empty types_absent for all later statements
            insns = Insn.convert_from_tokens(data.test_stmt_fqinsns[stmt_i])
            for insn in insns:
                if insn.is_method_insn():
                    try:
                        cid = self.rdloader.lookup_class(insn.get_owner())
                    except ClassResolutionException:
                        continue
                    try:
                        mid = self.rdloader.lookup_method(
                            cid, insn.get_method_namedesc(), insn.op
                        )
                    except MethodResolutionException:
                        self.log_to_file(
                            f"Failed to analyze method invocation insn {insn}"
                        )
                        continue
                    if mid < 0:
                        continue
                    if mid == data.focal_mid:
                        types_absent = set()

    def collect_last_called_methods(self, data: Data, max_tok: int = 200):
        data.last_called_methods = [None]

        for stmt_i in range(len(data.test_stmts)):
            # refresh the detecting of last called method per statement
            last_called_method = None
            insns = Insn.convert_from_tokens(data.test_stmt_fqinsns[stmt_i])
            for insn in insns:
                if insn.is_method_insn():
                    try:
                        cid = self.rdloader.lookup_class(insn.get_owner())
                    except ClassResolutionException:
                        continue
                    cs = self.rdloader.get_class(cid)
                    if cs.scope not in {Scope.APP, Scope.TEST}:
                        continue
                    try:
                        mid = self.rdloader.lookup_method(
                            cid, insn.get_method_namedesc(), insn.op
                        )
                    except MethodResolutionException:
                        self.log_to_file(
                            f"Failed to analyze method invocation insn {insn}"
                        )
                        continue
                    if mid < 0:
                        continue
                    ms = self.rdloader.get_method(mid)
                    if ms.ast is not None:
                        num_toks = ms.ast.size(count_nonterminal=False)
                        if num_toks > max_tok:
                            self.log_to_file(
                                f"Ignoring a very long method call {ms.namedesc} ({data.proj_name=}, {data.id=}, {num_toks} toks > {max_tok})"
                            )
                            continue
                        last_called_method = ms.ast
            data.last_called_methods.append(last_called_method)

    def collect_similar_stmts_per_project(
        self,
        max_method: int = 10_000,
        max_tok: int = 400,
        window_size: int = 2,
    ):
        # find all APP methods in the project, and collect prev_stmts - next_stmt data
        ds: List[Tuple[List[str], AST]] = []
        visitor = CollectSSVisitor()
        cnt_method = 0
        for cs in self.rdloader.all_classes:
            if cs.scope != Scope.APP:
                continue

            for mid in cs.methods:
                ms = self.rdloader.get_method(mid)
                if ms.bytecode is None or ms.ast is None:
                    continue

                if ms.ast.size(count_nonterminal=False) > max_tok:
                    self.log_to_file(
                        f"Ignoring a very long method call {ms.namedesc} ({cs.name=}, {mid=}, {ms.ast.size(count_nonterminal=False)} toks > {max_tok})"
                    )
                    continue

                # collect all prev_stmts - next_stmt data for this method
                visitor_ctx = CollectSSVisitorContext(window_size=window_size)
                visitor.visit(ms.ast, visitor_ctx)
                ds += visitor_ctx.collected

                cnt_method += 1
                if cnt_method >= max_method:
                    break
            if cnt_method >= max_method:
                self.log_to_file(
                    f"Ignoring all remaining methods because of already indexed {cnt_method} methods"
                )
                break

        if len(ds) == 0:
            self.ss_bm25 = None
            self.ss_docs = None
            self.ss_tgts = None
        else:
            # fit BM25 model
            self.ss_bm25 = BM25(
                vectorizer_kwargs=dict(
                    lowercase=False,
                    tokenizer=lambda s: [
                        st.lower() for t in s for st in RE_SUBTOKENIZER.split(t)
                    ],
                )
            )
            self.ss_docs = [d[0] for d in ds]
            self.ss_tgts = [d[1] for d in ds]
            self.ss_bm25.fit(self.ss_docs)

    def collect_similar_stmts(self, data: Data, window_size: int = 2):
        data.similar_stmts = []

        for stmt_i in range(len(data.test_stmts)):
            if stmt_i == 0:
                # first statement has no similar_stmt information
                data.similar_stmts.append(None)
                continue
            elif self.ss_bm25 is None:
                # no similar_stmt information available
                data.similar_stmts.append(None)
                continue

            # collect window_size previous statements
            query = sum(
                [
                    stmt.get_tokens()
                    for stmt in data.test_stmts[max(stmt_i - window_size, 0) : stmt_i]
                ],
                [],
            )

            # search using the BM25 model
            scores = self.ss_bm25.transform(query, self.ss_docs)
            match = scores.argmax().item()
            norm_score = scores[match].item() / len(query)
            # print(f"{stmt_i=}")
            # print(f"{query=}")
            # print(f"{self.ss_docs[match]=}")
            # print(f"{data.test_stmts[stmt_i].get_tokens()=}")
            # print(f"{self.ss_tgts[match].get_tokens()=}")
            # print(f"{match=}, {norm_score=}")

            data.similar_stmts.append((norm_score, self.ss_tgts[match]))

    def analyze(
        self,
        pretrained_tokenizer: Union[su.arg.RPath, str],
        out_dir: su.arg.RPath,
        eval_locs_file: su.arg.RPath,
        preds_dir: Optional[su.arg.RPath] = None,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if isinstance(pretrained_tokenizer, Path):
            pretrained_tokenizer = os.path.relpath(pretrained_tokenizer, Path.cwd())
        tokenizer: SubtokenizerBPE = SubtokenizerBPE(
            AutoTokenizer.from_pretrained(pretrained_tokenizer)
        )

        eval_locs: List[Tuple[str, int]] = su.io.load(eval_locs_file)

        # load dataset
        dataset = load_dataset(self.data_dir, clz=Data)
        indexed_dataset = {d.id: d for d in dataset}

        # potentially load predictions
        if preds_dir is not None:
            preds: List[Prediction] = su.io.load(
                preds_dir / "preds.jsonl", clz=Prediction
            )
            assert len(preds) == len(eval_locs)

        # prepare the output directory
        su.io.mkdir(out_dir)
        su.io.dump(out_dir / ".gitignore", "all.csv\n", su.io.Fmt.txt)

        # create a data frame with all data to be analyzed
        records = []
        for eval_loc_i, (data_id, stmt_i) in enumerate(
            tqdm(eval_locs, desc="Analyzing")
        ):
            d = indexed_dataset[data_id]
            record = {
                "data_id": d.id,
                "stmt_i": stmt_i,
                "proj_name": d.proj_name,
                "test_mkey": d.test_mkey,
                "focal_mkey": d.focal_mkey,
            }

            if preds_dir is not None:
                record.update(preds[eval_loc_i].metrics)
                baseline_t = preds[eval_loc_i].topk[0]["toks"]
                # identify degeneration cases: repeating any subtoken for >= 10 times
                record["degeneration"] = (
                    True
                    if collections.Counter(baseline_t).most_common(1)[0][1] >= 10
                    else False
                )
            else:
                baseline_t = []

            basic_t = d.focalm + d.test_sign + sum(d.test_stmts[:stmt_i], [])
            gold_t = d.test_stmts[stmt_i]
            basic_st = tokenizer.toks2esubtoks(basic_t)
            gold_st = tokenizer.toks2esubtoks(gold_t)
            baseline_st = tokenizer.toks2esubtoks(baseline_t)
            basic_stu = list(set(basic_st))
            gold_stu = list(set(gold_st))
            baseline_stu = list(set(baseline_st))
            miss_stu = list(set(gold_st) - set(basic_st))
            error_stu = list(set(gold_st) - set(baseline_st))
            record.update(
                {
                    "st:basic": len(basic_st),
                    "text_st:basic": "".join(tokenizer.esubtok2subtok(basic_st)),
                    "st:gold": len(gold_st),
                    "text_st:gold": "".join(tokenizer.esubtok2subtok(gold_st)),
                    "st:baseline": len(baseline_st),
                    "text_st:baseline": "".join(tokenizer.esubtok2subtok(baseline_st)),
                    "stu:basic": len(basic_stu),
                    "text_stu:basic": " ".join(basic_stu),
                    "stu:gold": len(gold_stu),
                    "text_stu:gold": " ".join(gold_stu),
                    "stu:baseline": len(baseline_stu),
                    "text_stu:baseline": " ".join(baseline_stu),
                    "stu:miss": len(miss_stu),
                    "text_stu:miss": " ".join(miss_stu),
                    "stu:error": len(error_stu),
                    "text_stu:error": " ".join(error_stu),
                }
            )

            for f in [
                "types_local",
                "types_absent",
                "fields_set",
                "fields_notset",
                "setup_method",
                "teardown_method",
                "setup_teardown",
            ]:
                if f in ["types_local", "types_absent", "fields_set", "fields_notset"]:
                    field_t = getattr(d, f)[stmt_i]
                elif f in ["setup_method", "teardown_method"]:
                    field_t = getattr(d, f)
                elif f == "setup_teardown":
                    field_t = d.setup_method + d.teardown_method
                else:
                    raise Exception(f"Unknown field: {f}")

                field_st = tokenizer.toks2esubtoks(field_t)
                field_stu = list(set(field_st))
                field_present = 1 if len(field_stu) > 0 else 0

                record.update(
                    {
                        f"st:{f}": len(field_st),
                        f"text_st:{f}": "".join(tokenizer.esubtok2subtok(field_st)),
                        f"stu:{f}": len(field_stu),
                        f"text_stu:{f}": " ".join(field_stu),
                        f"present:{f}": field_present,
                        f"o_basic:{f}": len([x for x in field_stu if x in basic_stu]),
                        f"o_gold:{f}": len([x for x in field_stu if x in gold_stu]),
                        f"o_baseline:{f}": len(
                            [x for x in field_stu if x in baseline_stu]
                        ),
                        f"o_miss:{f}": len([x for x in field_stu if x in miss_stu]),
                        f"o_error:{f}": len([x for x in field_stu if x in error_stu]),
                    }
                )
                record.update(
                    {
                        f"fo_basic:{f}": record[f"o_basic:{f}"] / record[f"stu:basic"]
                        if record[f"stu:basic"] > 0
                        else np.NaN,
                        f"fo_gold:{f}": record[f"o_gold:{f}"] / record[f"stu:gold"]
                        if record[f"stu:gold"] > 0
                        else np.NaN,
                        f"fo_baseline:{f}": record[f"o_baseline:{f}"]
                        / record[f"stu:baseline"]
                        if record[f"stu:baseline"] > 0
                        else np.NaN,
                        f"fo_miss:{f}": record[f"o_miss:{f}"] / record[f"stu:miss"]
                        if record[f"stu:miss"] > 0
                        else np.NaN,
                        f"fo_error:{f}": record[f"o_error:{f}"] / record[f"stu:error"]
                        if record[f"stu:error"] > 0
                        else np.NaN,
                    }
                )

            records.append(record)

        # dump this data frame (locally stored)
        df = pd.DataFrame.from_records(records)
        df.to_csv(out_dir / "all.csv")

        # generate some summary information
        df_summary = df.describe().transpose()
        df_summary.to_csv(out_dir / "summary.csv")

        # subsets of data frame to be further manually checked
        # 10 random cases
        df.sample(min(10, len(df))).to_json(
            out_dir / "samples-random.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )

        if preds_dir is not None:
            # 10 degeneration cases
            df_degen = df[df["degeneration"]]
            print(f"Degeneration cases: {len(df_degen)}/{len(df)}")
            df_degen.sample(min(10, len(df_degen))).to_json(
                out_dir / "samples-degen.json",
                orient="records",
                indent=2,
                force_ascii=False,
            )

            # 10 cases with lowest scores (edit-sim), except for degeneration cases
            df[~df["degeneration"]].sort_values(by="edit-sim", ascending=True)[
                :10
            ].to_json(
                out_dir / "samples-worst.json",
                orient="records",
                indent=2,
                force_ascii=False,
            )


# ----- auxiluary classes for collect similar stmts -----


@dataclasses.dataclass
class CollectSSNode:
    stmt: AST = None
    this_v: scipy.sparse.csr_matrix = None
    mname_v: scipy.sparse.csr_matrix = None
    prevs: Set["CollectSSNode"] = dataclasses.field(default_factory=set)
    prev_vs: List[scipy.sparse.csr_matrix] = dataclasses.field(default_factory=list)

    @functools.cached_property
    def stmt_doc(self) -> str:
        return " ".join(self.stmt.get_tokens())

    def __hash__(self) -> int:
        # simply use the object id to save time
        return id(self)


@dataclasses.dataclass
class CollectSSVisitorContext:
    # the maximum number of previous statements to look at
    window_size: int = 3
    # previous statements (at most window_size) at the same control flow level
    prev_stmts: Deque[List[str]] = None
    # all collected prev_stmts - next_stmt data
    collected: List[Tuple[List[str], AST]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.prev_stmts is None:
            self.prev_stmts = collections.deque(maxlen=self.window_size)

    def clone_for_inner_level(self) -> "CollectSSVisitorContext":
        return dataclasses.replace(
            self, prev_stmts=collections.deque(maxlen=self.window_size)
        )


class CollectSSVisitor(Visitor[CollectSSVisitorContext, None]):
    def record_terminal_stmt(self, node: AST, context: CollectSSVisitorContext):
        if len(context.prev_stmts) != 0:
            context.collected.append((sum(context.prev_stmts, []), node))
        context.prev_stmts.append(node.get_tokens())

    def visit_AssertStmt(self, node: AST, context: CollectSSVisitorContext):
        self.record_terminal_stmt(node, context)

    def visit_ExplicitConstructorInvocationStmt(
        self, node: AST, context: CollectSSVisitorContext
    ):
        self.record_terminal_stmt(node, context)

    def visit_ExpressionStmt(self, node: AST, context: CollectSSVisitorContext):
        self.record_terminal_stmt(node, context)

    def visit_ReturnStmt(self, node: AST, context: CollectSSVisitorContext):
        self.record_terminal_stmt(node, context)
        # control flow is interrupted
        context.prev_stmts.clear()

    def visit_ThrowStmt(self, node: AST, context: CollectSSVisitorContext):
        self.record_terminal_stmt(node, context)
        # control flow is interrupted
        context.prev_stmts.clear()

    def visit_LocalClassDeclarationStmt(
        self, node: AST, context: CollectSSVisitorContext
    ):
        # not recorded
        # visit the inner code of it
        self.default_visit(node, context.clone_for_inner_level())

    def visit_BreakStmt(self, node: AST, context: CollectSSVisitorContext):
        # not recorded
        # control flow is interrupted
        context.prev_stmts.clear()

    def visit_ContinueStmt(self, node: AST, context: CollectSSVisitorContext):
        # not recorded
        # control flow is interrupted
        context.prev_stmts.clear()

    def visit_IfStmt(self, node: AST, context: CollectSSVisitorContext):
        branches = [c for c in node.children if c.ast_type in Consts.asts_stmt]
        assert 1 <= len(branches) <= 2
        for branch in branches:
            self.visit(branch, context.clone_for_inner_level())

    def visit_WhileStmt(self, node: AST, context: CollectSSVisitorContext):
        self.default_visit(node, context.clone_for_inner_level())

    def visit_DoStmt(self, node: AST, context: CollectSSVisitorContext):
        self.default_visit(node, context.clone_for_inner_level())

    def visit_ForStmt(self, node: AST, context: CollectSSVisitorContext):
        self.default_visit(node, context.clone_for_inner_level())

    def visit_ForEachStmt(self, node: AST, context: CollectSSVisitorContext):
        self.default_visit(node, context.clone_for_inner_level())

    def visit_SwitchEntry(self, node: AST, context: CollectSSVisitorContext):
        self.default_visit(node, context.clone_for_inner_level())

    def visit_TryStmt(self, node: AST, context: CollectSSVisitorContext):
        blocks = [c for c in node.children if c.ast_type == Consts.ast_block_stmt]
        assert 1 <= len(blocks) <= 2
        try_block = blocks[0]
        finally_block = blocks[1] if len(blocks) == 2 else None
        catch_clauses = [
            c for c in node.children if c.ast_type == Consts.ast_catch_clause
        ]

        # try block
        self.default_visit(try_block, context.clone_for_inner_level())
        # each catch clause
        for catch_clause in catch_clauses:
            self.default_visit(catch_clause, context.clone_for_inner_level())
        # finally block, if that's here
        if finally_block is not None:
            self.default_visit(finally_block, context.clone_for_inner_level())


# ----------


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(ExtendedDataCollector, as_positional=False)
