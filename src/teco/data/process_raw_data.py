import collections
import dataclasses
import random
import re
import traceback
from pathlib import Path
from typing import Counter, List, Optional, Tuple

import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr
from seutil.project import Project
from tqdm import tqdm

from teco.data.data import BASIC_FIELDS, Data
from teco.data.raw_data_loader import (
    ClassResolutionException,
    MethodResolutionException,
    RawDataLoader,
)
from teco.data.structures import ClassStructure, Consts, Insn, MethodStructure, Scope
from teco.data.utils import save_dataset
from teco.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


END = "END"


class PlanExtractionException(Exception):
    def __init__(self, causes: List[str], stmt: str, insns: str, plan: str):
        self.causes = causes
        self.stmt = stmt
        self.insns = insns
        self.plan = plan

    def __str__(self):
        return f"causes: {self.causes};; stmt: {self.stmt};; insns: {self.insns};; plan: {self.plan};;"


class CodeMappingException(Exception):
    pass


@dataclasses.dataclass
class MethodUsage:
    scope: Scope = None
    is_assertion: bool = False
    has_code: bool = False
    is_abstract: bool = False
    mid: int = -1
    ms: MethodStructure = None


@dataclasses.dataclass
class ParseContext:
    insns: List[Insn]
    insn_i: int = 0
    not_fully_handled: List[str] = dataclasses.field(default_factory=list)

    def lookahead(self, skip_label: bool = True, skip_goto: bool = False) -> Insn:
        insn_i = self.insn_i + 1
        while insn_i < len(self.insns):
            if self.insns[insn_i].op == "LABEL" and skip_label:
                insn_i += 1
            elif self.insns[insn_i].op == "GOTO" and skip_goto:
                insn_i += 1
            else:
                return self.insns[insn_i]
        return Insn(END)

    def consume(self, skip_label: bool = False):
        self.insn_i += 1
        while self.insn_i < len(self.insns) and (
            self.insns[self.insn_i].op == "LABEL" and skip_label
        ):
            self.insn_i += 1

    def peek(self) -> Insn:
        return self.insns[self.insn_i]

    def has_next(self) -> bool:
        return self.insn_i < len(self.insns)


class DataProcessor:

    re_badly_named_tests = re.compile(r"[tT][eE][sS][tT]_?\d*")

    def __init__(
        self,
        min_test_stmt: int = 1,
        max_test_stmt: int = 20,
        min_focalm_tok: int = 1,
        max_focalm_tok: int = 200,
        max_test_focalm_tok: int = 400,
        max_tok_per_stmt: int = 100,
        max_data_per_proj: int = 10_000,
        min_data_per_proj: int = 1,
        min_star: int = 0,
        remove_badly_named_tests: bool = True,
        remove_irregular_tests: bool = True,
        seed: Optional[int] = None,
    ):
        self.config = {
            k: v for k, v in locals().items() if k not in {"self", "__class__"}
        }

        self.filter_counter: Counter[str] = collections.Counter()
        for k in self.config:
            if k == "seed":
                continue
            self.filter_counter[k] = 0
        self.filter_counter["missing_code"] = 0
        self.filter_counter["cannot_locate_focalm"] = 0
        self.filter_counter["no_data"] = 0
        self.filter_counter["code_mapping_error"] = 0
        self.filter_counter["insn_simplify_warning"] = 0
        self.filter_counter["has_control_flow"] = 0
        self.filter_counter["has_invokedynamic"] = 0

        # data counter
        self.data_id: int = 0

        if seed is not None:
            random.seed(seed)

        # raw data loader
        self.rdloader = RawDataLoader(indexed=True)

    def start_collecting(self):
        self.data_id = 0

    def finish_collecting_project(self, dataset: List[Data]):
        for data in dataset:
            data.id = f"csn-{self.data_id}"
            self.data_id += 1
        if len(dataset) > 0:
            save_dataset(self.out_dir, dataset, only=BASIC_FIELDS, append=True)

    def log_for_project(self, name: str, msg: str):
        su.io.dump(
            self.log_dir / "repo" / f"{name}.txt", [msg], su.io.Fmt.txtList, append=True
        )

    def process(
        self,
        repos_file: Path_fr,
        raw_data_dir: Path_drw,
        out_dir: Path_dc,
        project_names: Optional[List[str]] = None,
        skip_project_names: Optional[List[str]] = None,
    ):
        repos_file = Path(repos_file.abs_path)
        raw_data_dir = Path(raw_data_dir.abs_path)
        self.out_dir = Path(out_dir.abs_path)

        su.io.mkdir(self.out_dir, fresh=True)
        self.log_dir = self.out_dir / "log"
        su.io.mkdir(self.log_dir, fresh=True)

        su.io.dump(self.out_dir / "config.json", self.config, su.io.Fmt.jsonNoSort)

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        projects.sort(key=lambda p: p.full_name)

        # limit to user specified projects
        if project_names is not None:
            projects = [p for p in projects if p.full_name in project_names]
            logger.info(
                f"Selected {len(projects)} projects: {[p.full_name for p in projects]}"
            )
        if skip_project_names is not None:
            new_projects = [
                p for p in projects if p.full_name not in skip_project_names
            ]
            logger.info(f"Skipped {len(projects) - len(new_projects)} projects")
            projects = new_projects

        self.start_collecting()
        pbar = tqdm(total=len(projects))

        # load jre data
        self.rdloader.load_jre_data(raw_data_dir / "jre")

        # process data for each project
        invalid = []
        ignored = []
        valid = []
        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (+{len(valid)} x{len(invalid)} -{len(ignored)} T{self.data_id})"
            )
            project_dir = raw_data_dir / p.full_name

            try:
                if p.data["stars"] < self.config["min_star"]:
                    ignored.append(p.full_name)
                    self.filter_counter["min_star"] += 1
                    self.log_for_project(
                        p.full_name,
                        f"Ignored: {p.data['stars']} < {self.config['min_star']}",
                    )
                    continue

                dataset = self.process_raw_data_project(p.full_name, project_dir)
                if len(dataset) == 0:
                    ignored.append(p.full_name)
                    self.filter_counter["no_data"] += 1
                    self.log_for_project(
                        p.full_name, "No data collected (after filtering)"
                    )
                    continue

                # limit number of data per project
                if len(dataset) < self.config["min_data_per_proj"]:
                    ignored.append(p.full_name)
                    self.filter_counter["min_data_per_proj"] += 1
                    self.log_for_project(
                        p.full_name,
                        f"Too few data collected (after filtering): {len(dataset)}",
                    )
                    continue
                if len(dataset) > self.config["max_data_per_proj"]:
                    self.filter_counter["max_data_per_proj"] += (
                        len(dataset) - self.config["max_data_per_proj"]
                    )
                    # randomly select a subset of data
                    ids = list(range(len(dataset)))
                    random.shuffle(ids)
                    ids = sorted(ids[: self.config["max_data_per_proj"]])
                    dataset = [dataset[i] for i in ids]

                self.finish_collecting_project(dataset)
                valid.append(p.full_name)
            except KeyboardInterrupt:
                input(
                    "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                )
                logger.warning(
                    f"Project {p.full_name} invalid due to: User interrupted"
                )
                invalid.append(p.full_name)
            except Exception as e:
                logger.info(
                    f"Project {p.full_name} invalid due to: {type(e)}; {traceback.format_exc()}"
                )
                invalid.append(p.full_name)
            finally:
                pbar.update(1)
        pbar.set_description(
            f"Finished (+{len(valid)} x{len(invalid)} -{len(ignored)} T{self.data_id})"
        )
        pbar.close()

        # save metrics
        su.io.dump(self.out_dir / "invalid_projects.txt", invalid, su.io.Fmt.txtList)
        su.io.dump(self.out_dir / "valid_projects.txt", valid, su.io.Fmt.txtList)
        su.io.dump(self.out_dir / "ignored_projects.txt", ignored, su.io.Fmt.txtList)
        su.io.dump(
            self.out_dir / "filter_counter.json",
            self.filter_counter,
            su.io.Fmt.jsonNoSort,
        )

    def process_raw_data_project(self, proj_name: str, proj_dir: Path) -> List[Data]:
        # load project data
        self.rdloader.load_project_data(proj_dir)

        # find test methods
        test_methods: List[MethodStructure] = [
            ms for ms in self.rdloader.methods if ms.is_test
        ]

        # collect dataset
        dataset: List[Data] = []
        for test_ms in test_methods:
            test_ms: MethodStructure

            # check test quality
            if self.config["remove_badly_named_tests"]:
                if self.re_badly_named_tests.fullmatch(test_ms.name) is not None:
                    self.filter_counter["remove_badly_named_tests"] += 1
                    self.log_for_project(
                        proj_name, f"Removed badly named test: {test_ms.namedesc}"
                    )
                    continue

            # check test regularity
            if self.config["remove_irregular_tests"]:
                if len(test_ms.ptypes) > 0:
                    # likely junit 3 parameterized tests left overs
                    self.filter_counter["remove_irregular_tests"] += 1
                    self.log_for_project(
                        proj_name,
                        f"Removed irregular test: {test_ms.namedesc}: has parameters",
                    )
                    continue
                if test_ms.rtype != "void":
                    self.filter_counter["remove_irregular_tests"] += 1
                    self.log_for_project(
                        proj_name,
                        f"Removed irregular test: {test_ms.namedesc}: return type not void",
                    )
                    continue

            # make sure test ast & bytecode is collected
            if test_ms.ast is None or test_ms.bytecode is None:
                self.filter_counter["missing_code"] += 1
                self.log_for_project(proj_name, f"Missing code: {test_ms.namedesc}")
                continue

            # check test regularity with looking at ast
            if self.config["remove_irregular_tests"]:
                has_local_class_declaration_stmt = False
                for node in test_ms.ast.traverse():
                    if node.ast_type == Consts.ast_local_class_declaration_stmt:
                        has_local_class_declaration_stmt = True
                        self.filter_counter["remove_irregular_tests"] += 1
                        self.log_for_project(
                            proj_name,
                            f"Removed irregular test: {test_ms.namedesc}: has local class declaration",
                        )
                        break
                if has_local_class_declaration_stmt:
                    continue

            # locate focal method
            focalm, log = self.locate_focalm(test_ms)
            if focalm is None:
                self.filter_counter["cannot_locate_focalm"] += 1
                self.log_for_project(
                    proj_name,
                    f"Cannot locate focal method: {test_ms.namedesc}: {log}",
                )
                continue
            else:
                # self.log_for_project(
                #     proj_name,
                #     f"INFO: Located focal method: {test_ms.namedesc} is {focalm.name}: {log}",
                # )
                pass

            # create data instance
            data = Data()

            data.proj_name = proj_name

            data.test_mid = test_ms.id
            data.test_cid = test_ms.clz
            data.test_mkey = (
                f"{self.rdloader.get_class(test_ms.clz).iname}#{test_ms.namedesc}"
            )

            data.focal_mid = focalm.id
            data.focal_cid = focalm.clz
            data.focal_mkey = (
                f"{self.rdloader.get_class(focalm.clz).iname}#{focalm.namedesc}"
            )

            # collect tokens
            try:
                ret = self.collect_tokens(data, test_ms)
                if len(ret["simplify_insn_warning"]) != 0:
                    self.filter_counter["insn_simplify_warning"] += 1
                    self.log_for_project(
                        proj_name,
                        f"insns simplify warning: {test_ms.namedesc}: {ret['simplify_insn_warning']}",
                    )
            except CodeMappingException:
                self.filter_counter["code_mapping_error"] += 1
                self.log_for_project(
                    proj_name, f"Code mapping error: {test_ms.namedesc}"
                )
                continue
            self.collect_focalm_tokens(data, focalm)

            # filter out data with too few/many stmts/tokens
            if len(data.test_stmts) < self.config["min_test_stmt"]:
                self.filter_counter["min_test_stmt"] += 1
                self.log_for_project(
                    proj_name,
                    f"Too few test stmts: {test_ms.namedesc}: {len(data.test_stmts)}",
                )
                continue
            if len(data.test_stmts) > self.config["max_test_stmt"]:
                self.filter_counter["max_test_stmt"] += 1
                self.log_for_project(
                    proj_name,
                    f"Too many test stmts: {test_ms.namedesc}: {len(data.test_stmts)}",
                )
                continue
            if len(data.focalm_toks) < self.config["min_focalm_tok"]:
                self.filter_counter["min_focalm_tok"] += 1
                self.log_for_project(
                    proj_name,
                    f"Too few focalm tokens: {test_ms.namedesc}: {len(data.focalm_toks)}",
                )
                continue
            if len(data.focalm_toks) > self.config["max_focalm_tok"]:
                self.filter_counter["max_focalm_tok"] += 1
                self.log_for_project(
                    proj_name,
                    f"Too many focalm tokens: {test_ms.namedesc}: {len(data.focalm_toks)}",
                )
                continue
            all_toks = sum(
                [data.focalm_toks, data.test_sign_toks] + data.test_stmt_toks, []
            )
            if len(all_toks) > self.config["max_test_focalm_tok"]:
                self.filter_counter["max_test_focalm_tok"] += 1
                self.log_for_project(
                    proj_name,
                    f"Too many test+focalm tokens: {test_ms.namedesc}: {len(all_toks)}",
                )
                continue
            if any(
                len(stmt) > self.config["max_tok_per_stmt"]
                for stmt in data.test_stmt_toks
            ):
                self.filter_counter["max_tok_per_stmt"] += 1
                self.log_for_project(
                    proj_name,
                    f"Too many tokens per test stmt: {test_ms.namedesc}",
                )
                continue

            if any(len(cf) > 0 for cf in ret["test_cfs"] + ret["test_cf_insns"]):
                self.filter_counter["has_control_flow"] += 1
                self.log_for_project(proj_name, f"Has control flow: {test_ms.namedesc}")
                continue
            if any(t == "INVOKEDYNAMIC" for insn in data.test_stmt_insns for t in insn):
                self.filter_counter["has_invokedynamic"] += 1
                self.log_for_project(
                    proj_name, f"Has invokedynamic: {test_ms.namedesc}"
                )
                continue

            dataset.append(data)

        return dataset

    def get_methods_used_by(self, ms: MethodStructure) -> List[MethodUsage]:
        if ms.bytecode is None:
            return []

        methods = []
        for insn in ms.bytecode.get_ordered_insns():
            if insn.is_method_insn():
                try:
                    cid = self.rdloader.lookup_class(insn.get_owner())
                except ClassResolutionException:
                    continue
                cs = self.rdloader.get_class(cid)

                # resolve the method
                try:
                    mid = self.rdloader.lookup_method(
                        cid, insn.get_method_namedesc(), insn.op
                    )
                except MethodResolutionException:
                    # no need to really care about that at this point
                    continue

                ms = self.rdloader.get_method(mid)

                # check if method is assertion (matching method name only, to support user defined assertion methods)
                is_assertion = False
                for prefix in ["assert", "fail"]:
                    if ms.name.startswith(prefix):
                        is_assertion = True
                        break

                methods.append(
                    MethodUsage(
                        scope=cs.scope,
                        is_assertion=is_assertion,
                        has_code=(ms.ast is not None and ms.bytecode is not None),
                        is_abstract=ms.is_abstract,
                        mid=mid,
                        ms=ms,
                    )
                )
        return methods

    def locate_focalm(
        self, test_ms: MethodStructure
    ) -> Tuple[Optional[MethodStructure], str]:
        # collect the methods used in the test by traversing the bytecode (as the call graph collected in the raw data was inaccurate)
        methods = self.get_methods_used_by(test_ms)
        log = f"{len(methods)} methods used; "

        in_app = [
            i for i, m in enumerate(methods) if m.scope == Scope.APP and m.has_code
        ]
        log += f"{len(in_app)} in APP with ast; "
        if len(in_app) == 0:
            log += "other methods used: " + ", ".join(
                f"[{m.mid}] {m.ms.name} scope={m.scope} has_ast={m.has_code}"
                for m in methods
            )
            return None, log

        # 1. the only mut
        if len(in_app) == 1:
            return methods[in_app[0]].ms, log

        before_assertion = []
        for i, m in enumerate(methods):
            if m.is_assertion:
                break
            before_assertion.append(i)
        log += f"{len(before_assertion)} before assertion; "

        # 2. in focal class
        focalc = self.locate_focalc(test_ms)
        if focalc is not None:
            in_focalc = [i for i in in_app if methods[i].ms.clz == focalc.id]
            log += f"{len(in_focalc)} in focalc ({focalc.name}); "

            # 2.1. the only mut ~
            if len(in_focalc) == 1:
                return methods[in_focalc[0]].ms, log

            # 2.2. the last mut ~ before the first assertion
            if len(in_focalc) > 1:
                in_focalc_before_assertion = [
                    i for i in in_focalc if i in before_assertion
                ]
                log += f"{len(in_focalc_before_assertion)} in focalc before assertion; "
                if len(in_focalc_before_assertion) > 0:
                    return methods[in_focalc_before_assertion[-1]].ms, log

        # 3. any mut before the first assertion
        in_app_before_assertion = [i for i in in_app if i in before_assertion]
        log += f"{len(in_app_before_assertion)} in APP before assertion; "
        if len(in_app_before_assertion) > 0:
            return methods[in_app_before_assertion[-1]].ms, log

        # 4. (last resort) the last mut in the collected list
        log += "last resort; "
        return methods[in_app[-1]].ms, log

    def locate_focalc(self, test_cs: ClassStructure) -> Optional[ClassStructure]:
        components = test_cs.name.split(".")
        pkl = ".".join(components[:-1])
        cname = components[-1]
        if cname.endswith("Test"):
            cname = cname[:-4]
        elif cname.startswith("Test"):
            cname = cname[4:]

        fqcname = pkl + "." + cname
        try:
            cid = self.rdloader.lookup_class(Insn.class_q2iname(fqcname))
        except ClassResolutionException:
            return None
        if (cs := self.rdloader.get_class(cid)).scope == Scope.APP:
            return cs
        else:
            return None

    re_cdesc = re.compile(
        r"L(([a-zA-Z_][a-zA-Z_0-9]*)/)+(?P<canonical>[a-zA-Z_$][a-zA-Z_$0-9]*);"
    )
    re_fqname = re.compile(
        r"(([a-zA-Z_][a-zA-Z_0-9]*)/)+(?P<canonical>[a-zA-Z_$][a-zA-Z_$0-9]*)"
    )

    def collect_tokens(self, data: Data, test_ms: MethodStructure) -> dict:
        ret = {"simplify_insn_warning": "", "test_cfs": [], "test_cf_insns": []}

        data.test_sign = []
        data.test_stmts = []
        data.test_cfs = []

        # collect sign ast
        data.test_sign = test_ms.ast.get_sign()

        # collect stmt & cf asts; record their line numbers
        front_lineno = 0
        last_was_stmt = False
        lineno2stmt_i = {}
        lineno2cf_i = {}
        front_cf = []
        # read each statement in test body, excluding opening and closing brackets
        for n_out in test_ms.ast.get_body().children[1:-1]:
            for n in n_out.traverse(lambda x: x.ast_type in Consts.asts_terminal_stmt):
                if n.ast_type in Consts.asts_terminal_stmt:
                    # terminal statement
                    lineno_beg, lineno_end = n.get_lineno_range()
                    if lineno_beg <= front_lineno:
                        raise CodeMappingException("multiple statements on same line")
                    front_lineno = lineno_end
                    for lineno in range(lineno_beg, lineno_end + 1):
                        lineno2stmt_i[lineno] = len(data.test_stmts)
                    ret["test_cfs"].append(front_cf)
                    front_cf = []
                    data.test_stmts.append(n)
                    last_was_stmt = True
                elif n.is_terminal():
                    # control flow tokens in between terminal statements
                    lineno_beg, lineno_end = n.get_lineno_range()
                    if lineno_beg < front_lineno:
                        raise CodeMappingException("multiple statements on same line")
                    elif lineno_beg == front_lineno and last_was_stmt:
                        raise CodeMappingException("multiple statements on same line")
                        # it is ok to have multiple control flow tokens on the same line
                    front_lineno = lineno_end
                    for lineno in range(lineno_beg, lineno_end + 1):
                        lineno2cf_i[lineno] = len(ret["test_cfs"])
                    front_cf.append(n.tok)
                    last_was_stmt = False
                # other non-terminals
        # add any remaining control flow tokens
        ret["test_cfs"].append(front_cf)

        # collect instructions
        stmt_insns: List[List[Insn]] = [[] for _ in data.test_stmts]
        cf_insns: List[List[Insn]] = [[] for _ in ret["test_cfs"]]

        cur_lineno = None
        for i, insn in test_ms.bytecode.insns.items():
            if i in test_ms.bytecode.lnt:
                cur_lineno = test_ms.bytecode.lnt[i]
            if cur_lineno is None:
                raise CodeMappingException("no lineno for instruction")
            if cur_lineno in lineno2stmt_i:
                stmt_insns[lineno2stmt_i[cur_lineno]].append(insn)
            elif cur_lineno in lineno2cf_i:
                cf_insns[lineno2cf_i[cur_lineno]].append(insn)
            elif insn.op != Consts.op_return:
                # dangling instructions, due to some bug during collecting raw data
                # TODO: one bug is: when multiple classes with identical fully qualified names exist in a project (of multiple modules), only random one of the multiple .java and .class files is used, so there is a potential mismatch
                raise CodeMappingException(
                    "dangling instruction not matched to any source code"
                )

        # keep the fq version in data in case we need to use it later
        data.test_stmt_fqinsns = [Insn.convert_to_tokens(insns) for insns in stmt_insns]
        # but use the non-fq version by default
        data.test_stmt_insns = [
            [
                self.re_fqname.sub(
                    r"\g<canonical>", self.re_cdesc.sub(r"L\g<canonical>;", t)
                )
                for t in insns
            ]
            for insns in data.test_stmt_fqinsns
        ]
        ret["test_cf_insns"] = [Insn.convert_to_tokens(insns) for insns in cf_insns]

        # # get one-token actions
        # data.test_stmt_actions = []
        # for insns in stmt_insns:
        #     try:
        #         action = self.get_action(insns)
        #     except KeyboardInterrupt:
        #         raise
        #     except:
        #         logger.warning(
        #             f"failed to get action for {insns}: {traceback.format_exc()}"
        #         )
        #         action = Actions.unknown
        #     data.test_stmt_actions.append(action)

        return ret

    # def get_action(self, insns: List[Insn]) -> str:
    #     if len(insns) > 0:
    #         if (insns[-1].op in Consts.ops_store_insn) or (
    #             insns[-1].op == Consts.op_putfield
    #             and insns[0].op == Consts.op_aload
    #             and insns[0].operands[0] == "0"
    #         ):
    #             # check if we're initializing a new value or continuing a previous one
    #             using_prev_inputs = False
    #             last_insn = None
    #             for insn in insns[:-1]:
    #                 if (
    #                     insn.op in Consts.ops_load_insn and insn.operands[0] != "0"
    #                 ) or (
    #                     insn.op == Consts.op_getfield
    #                     and last_insn is not None
    #                     and last_insn.op == Consts.op_aload
    #                     and last_insn.operands[0] == "0"
    #                 ):
    #                     using_prev_inputs = True
    #                     break
    #                 last_insn = insn
    #             if using_prev_inputs:
    #                 return Actions.continue_input
    #             else:
    #                 return Actions.new_input
    #         elif insns[-1].op in Consts.ops_array_store_insn:
    #             return Actions.continue_input
    #         elif insns[-1].op in Consts.ops_method_insn or (
    #             insns[-1].op in {Consts.op_pop, Consts.op_pop2}
    #             and insns[-2].op in Consts.ops_method_insn
    #         ):
    #             if insns[-1].op in Consts.ops_method_insn:
    #                 mname = insns[-1].get_method_name()
    #             else:
    #                 mname = insns[-2].get_method_name()

    #             # check if it's an assertion method
    #             if mname.startswith("assert") or mname.startswith("fail"):
    #                 return Actions.assertion
    #             else:
    #                 return Actions.invoke
    #     else:
    #         return Actions.empty
    #     return Actions.unknown

    def collect_focalm_tokens(self, data: Data, focalm_ms: MethodStructure):
        data.focalm = focalm_ms.ast


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(DataProcessor, as_positional=False)
