import atexit
import collections
import dataclasses
import os
import traceback
from typing import List, Optional

import seutil as su
from jsonargparse import CLI
from seutil.project import Project
from tqdm import tqdm

from teco.data.data import Data
from teco.data.maven import MavenProjectHelper
from teco.data.structures import Consts
from teco.data.tool import DataCollector, ensure_tool_versions
from teco.data.utils import load_dataset, save_dataset
from teco.exe.adhoc_runner_generator import AdHocRunnerGenerator
from teco.macros import Macros

logger = su.log.get_logger(__name__)


@dataclasses.dataclass
class ExtendedDataScope:
    name: str
    fields: List[str]
    deps: List[str]


EXTENDED_DATA_GROUPS = {
    "runtime_types_values": ExtendedDataScope(
        name="runtime_types_values",
        fields=["runtime_types_values_delta"],
        deps=["test_sign", "test_stmts", "proj_name", "test_mkey"],
    ),
}


class ExtendRuntimeDataCollector:
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

    CLONE_TIMEOUT = 300
    COMPILE_TIMEOUT = 300
    COLLECT_TIMEOUT = 100

    def __init__(self, timeout_per_test: int = 5, max_value_char: int = 100):
        self.config = {
            k: v for k, v in locals().items() if k not in {"self", "__class__"}
        }

    def collect(
        self,
        name: str,
        data_dir: su.arg.RPath,
        repos_file: su.arg.RPath,
        downloads_dir: su.arg.RPath = Macros.downloads_dir,
        temp_dir: Optional[su.arg.RPath] = None,
        begin_proj_i: Optional[int] = None,
        end_proj_i: Optional[int] = None,
        overwrite: bool = False,
        only_sets: List[str] = None,
    ):
        group = EXTENDED_DATA_GROUPS[name]

        self.downloads_dir = downloads_dir
        if temp_dir is None:
            temp_dir = su.io.mktmp_dir(prefix="teco")
        else:
            temp_dir = su.io.mktmp_dir(prefix="teco", dir=temp_dir)
        self.temp_dir = temp_dir

        # check tool versions
        ensure_tool_versions()

        # places to hold logging information
        if name == "runtime_types_values":
            log_file_name = "extra-runtime"
        else:
            logger.error(f"Unknown item to collect: {name}")
            raise RuntimeError(f"Unknown item to collect: {name}")

        self.cur_log_file = data_dir / "log" / f"{log_file_name}.txt"
        if not overwrite and self.cur_log_file.exists():
            for i in range(1, 100):
                self.cur_log_file = data_dir / "log" / f"{log_file_name}.{i}.txt"
                if not self.cur_log_file.exists():
                    break
        su.io.rm(self.cur_log_file)
        invalid_projects = []
        valid_count = 0
        invalid_count = 0

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        indexed_projects = {p.full_name: p for p in projects}

        # load the entire dataset
        fields = group.deps
        if not overwrite:
            fields += group.fields
        with tqdm(desc="loading dataset") as pbar:
            dataset = load_dataset(data_dir, clz=Data, only=fields, pbar=pbar)

        # group data by project
        pname2dataset = collections.defaultdict(list)
        for data in dataset:
            pname2dataset[data.proj_name].append(data)

        pnames = list(sorted(pname2dataset.keys()))

        # for proj_i, pname in enumerate(pnames):
        #     proj_dataset = pname2dataset[pname]
        #     data_ids = [int(data.id.split("-")[1]) for data in proj_dataset]
        #     print(proj_i, pname, min(data_ids), max(data_ids))
        # raise KeyboardInterrupt

        if begin_proj_i is not None or end_proj_i is not None:
            if begin_proj_i is None:
                begin_proj_i = 0
            if end_proj_i is None:
                end_proj_i = len(dataset)
            logger.warning(
                f"selecting a subset of the projects {begin_proj_i}:{end_proj_i}"
            )
            pnames = pnames[begin_proj_i:end_proj_i]

        if only_sets is not None:
            pnames = [
                p for p in pnames if indexed_projects[p].data["sources"] in only_sets
            ]

        # prepare the adhoc runner generator
        DataCollector.require_compiled()
        if name == "runtime_types_values":
            generator_main_class = "org.teco.AdHocRunnerGenerator"
        self.generator = AdHocRunnerGenerator(generator_main_class)
        self.generator.setup()
        atexit.register(self.generator.teardown)

        pbar = tqdm(
            total=sum(len(pname2dataset[pname]) for pname in pnames),
            desc="start collecting data",
        )
        for proj_i, pname in enumerate(pnames):
            proj_dataset = pname2dataset[pname]
            if pname not in indexed_projects:
                self.log_to_file(f"ERROR: proj#{proj_i} {pname} not in repos file")
                invalid_projects.append(pname)
                invalid_count += len(proj_dataset)
                pbar.update(len(proj_dataset))
                continue

            # prepare each project: clone, compile, get class path
            pbar.set_description(
                f"(+{valid_count}|-{invalid_count}|proj#{proj_i}) preparing project {pname}"
            )
            proj = indexed_projects[pname]
            try:
                proj_info = self.prepare_project(proj)
            except KeyboardInterrupt:
                input(
                    "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                )
                logger.warning(f"Processing failed for {pname}: User interrupted")
                invalid_projects.append(pname)
                invalid_count += len(proj_dataset)
                pbar.update(len(proj_dataset))
                continue
            except:
                self.log_to_file(
                    f"ERROR: failed to prepare project {pname}: {traceback.format_exc()}"
                )
                invalid_projects.append(pname)
                invalid_count += len(proj_dataset)
                pbar.update(len(proj_dataset))
                continue

            # collect for each data
            proj_success_count = 0
            for data in proj_dataset:
                pbar.set_description(
                    f"(+{valid_count}|-{invalid_count}|proj#{proj_i}) collecting data {data.test_mkey}"
                )
                try:
                    with su.TimeUtils.time_limit(
                        self.COLLECT_TIMEOUT + self.config["timeout_per_test"]
                    ):
                        if name == "runtime_types_values":
                            self.collect_types_values(data, proj_info)
                        else:
                            logger.error(f"Unknown item to collect: {name}")
                            raise RuntimeError(f"Unknown item to collect: {name}")
                    valid_count += 1
                    proj_success_count += 1
                except KeyboardInterrupt:
                    input(
                        "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current data and continue..."
                    )
                    logger.warning(
                        f"Processing failed for proj#{proj_i} {pname} data #{data.id} {data.test_mkey}: User interrupted"
                    )
                    invalid_count += 1
                    continue
                except:
                    # logger.warning(
                    #     f"WARNING: failed to collect for data #{data.id} {data.test_mkey}: {traceback.format_exc()}"
                    # )
                    # input(
                    #     "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current data and continue..."
                    # )
                    self.log_to_file(
                        f"WARNING: failed to collect for proj#{proj_i} {pname} data #{data.id} {data.test_mkey}: {traceback.format_exc()}"
                    )
                    invalid_count += 1
                    continue
                finally:
                    pbar.update(1)

            if proj_success_count == 0:
                self.log_to_file(
                    f"WARNING: no data collected for proj#{proj_i} {pname}"
                )
                invalid_projects.append(pname)

        pbar.close()

        if invalid_count > 0:
            self.log_to_file("=====")
            self.log_to_file(f"total invalid: {invalid_count}")
            self.log_to_file(f"invalid projects: {invalid_projects}")
            logger.warning(
                f"total invalid: {invalid_count}, total invalid projects: {len(invalid_projects)}"
            )

        # save collected data
        save_dataset(data_dir, dataset, only=group.fields)

        # clean up temp dir
        su.io.rmdir(self.temp_dir)

    def prepare_project(self, proj: Project) -> dict:
        # clone the project
        with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
            proj.clone(self.downloads_dir)
        proj.checkout(proj.data["sha"], forced=True)

        # compile all classes (including application and test classes)
        with su.TimeUtils.time_limit(self.COMPILE_TIMEOUT):
            with su.io.cd(proj.dir):
                rr = su.bash.run(f"mvn test-compile", timeout=self.COMPILE_TIMEOUT)
                if rr.returncode != 0:
                    # try to clean compile once again
                    su.bash.run(
                        f"mvn clean test-compile", 0, timeout=self.COMPILE_TIMEOUT
                    )

        # collect information about the project
        return {
            "proj_dir": proj.dir,
            "test_java_roots": MavenProjectHelper.get_test_java_roots(proj),
            "classpath": os.pathsep.join(
                [
                    MavenProjectHelper.get_app_class_path(proj),
                    MavenProjectHelper.get_test_class_path(proj),
                    MavenProjectHelper.get_dependency_classpath(proj),
                ]
            ),
        }

    def collect_types_values(self, data: Data, proj_info: dict):
        cname = data.test_mkey.split("#")[0].split("/")[-1]
        mname = data.test_mkey.split("#")[1].split("(")[0]

        # 1. locate the original file that contains the data
        # find the class in each test java root; expect exactly one match
        rel_path = data.test_mkey.split("#")[0] + ".java"
        found_roots = []
        for test_java_root in proj_info["test_java_roots"]:
            possible_file = proj_info["proj_dir"] / test_java_root / rel_path
            if possible_file.exists():
                found_roots.append(proj_info["proj_dir"] / test_java_root)
        if len(found_roots) == 0:
            raise RuntimeError(f"no original file found for {data.test_mkey}")
        elif len(found_roots) > 1:
            raise RuntimeError(f"multiple original files found for {data.test_mkey}")

        found_root = found_roots[0]
        src_path = found_root / rel_path

        # 2. modify this file to get an ad-hoc runner for the tests with additional prints
        # prepare a temp dir for putting the logs
        temp_dir = su.io.mktmp_dir(prefix=data.proj_name, dir=self.temp_dir)
        log_path = temp_dir / "log"
        su.io.mkdir(log_path)

        # generate the modified test at the temp directory with same package structure (with teco_ prefix)
        # test resources should be available on classpath, except for some cornel cases
        run_root = temp_dir / "java"
        pck_rel_path = "/".join(data.test_mkey.split("#")[0].split("/")[:-1])
        out_path = run_root / pck_rel_path / f"teco_{cname}.java"
        su.io.mkdir(out_path.parent)

        # call the generator (in Java)
        self.generator.generate(
            str(src_path.absolute()),
            cname,
            mname,
            str(out_path.absolute()),
            str(log_path.absolute()),
        )

        # 3. compile and run the ad-hoc runner
        classpath = os.pathsep.join(
            [proj_info["classpath"], DataCollector.adhoc_runner_jar]
        )
        with su.io.cd(run_root):
            # compile
            su.bash.run(f"javac -cp {classpath} {out_path}", 0)

            # run
            run_cname = pck_rel_path.replace("/", ".") + ".teco_" + cname
            run_cmds = []
            if "junit-4" in proj_info["classpath"]:
                # JUnit 4: first, try to use JUnit runner
                run_cmds.append(
                    f"java -cp .:{classpath} org.junit.runner.JUnitCore {run_cname}"
                )
            # use the ad-hoc runner
            run_cmds.append(f"java -cp .:{classpath} {run_cname}")

            for run_cmd in run_cmds:
                rr_test = su.bash.run(
                    run_cmd, timeout=self.config["timeout_per_test"], warn_nonzero=False
                )
                if rr_test.returncode == 0:
                    break
                # else:
                #     logger.warning(
                #         f"{run_cmd} failed, return code: {rr_test.returncode}\nstdout: {rr_test.stdout}\nstderr: {rr_test.stderr}\n"
                #     )

            # make sure we had a successful run
            if rr_test.returncode != 0:
                raise RuntimeError(
                    f"test failed with return code: {rr_test.returncode}\nstdout: {rr_test.stdout}\nstderr: {rr_test.stderr}\n"
                )

        # 4. collect results
        # print(f"- sign: {' '.join(data.test_sign.get_tokens())}")
        data.runtime_types_values_delta = []
        cur_types_values = {}
        for stmt_i in range(len(data.test_stmts) + 1):
            this_types_values = {}
            typevalue_file = log_path / f"typevalue-{stmt_i}"
            if typevalue_file.exists():
                for line in su.io.load(typevalue_file, su.io.Fmt.txtList):
                    name, type_, value = line.split(" ", 2)
                    this_types_values[name] = (type_, value)

            names_nochange = []
            for name in this_types_values:
                if (
                    name in cur_types_values
                    and this_types_values[name] == cur_types_values[name]
                ):
                    names_nochange.append(name)
                else:
                    cur_types_values[name] = this_types_values[name]

            for name in names_nochange:
                del this_types_values[name]

            data.runtime_types_values_delta.append(this_types_values)

            # print(f"- stmt#{stmt_i}")
            # print(f"  - runtime_types: {this_runtime_types}")
            # print(f"  - runtime_values: {this_runtime_values}")
            # if stmt_i < len(data.test_stmts):
            #     print(f"  - stmt: {' '.join(data.test_stmts[stmt_i].get_tokens())}")

        su.io.rm(out_path)
        su.io.rmdir(temp_dir)

    def collect_depth1_values(self, data: Data, proj_info: dict):
        cname = data.test_mkey.split("#")[0].split("/")[-1]
        mname = data.test_mkey.split("#")[1].split("(")[0]

        # 1. locate the original file that contains the data
        # find the class in each test java root; expect exactly one match
        rel_path = data.test_mkey.split("#")[0] + ".java"
        found_roots = []
        for test_java_root in proj_info["test_java_roots"]:
            possible_file = proj_info["proj_dir"] / test_java_root / rel_path
            if possible_file.exists():
                found_roots.append(proj_info["proj_dir"] / test_java_root)
        if len(found_roots) == 0:
            raise RuntimeError(f"no original file found for {data.test_mkey}")
        elif len(found_roots) > 1:
            raise RuntimeError(f"multiple original files found for {data.test_mkey}")

        found_root = found_roots[0]
        src_path = found_root / rel_path

        # 2. modify this file to get an ad-hoc runner for the tests with additional prints
        # prepare a temp dir for putting the logs
        temp_dir = su.io.mktmp_dir(prefix=data.proj_name, dir=self.temp_dir)
        log_path = temp_dir / "log"
        su.io.mkdir(log_path)

        # generate the modified test at the temp directory with same package structure (with teco_ prefix)
        # test resources should be available on classpath, except for some cornel cases
        run_root = temp_dir / "java"
        pck_rel_path = "/".join(data.test_mkey.split("#")[0].split("/")[:-1])
        out_path = run_root / pck_rel_path / f"teco_{cname}.java"
        su.io.mkdir(out_path.parent)

        # call the generator (in Java)
        self.generator.generate(
            str(src_path.absolute()),
            cname,
            mname,
            str(out_path.absolute()),
            str(log_path.absolute()),
        )

        # 3. compile and run the ad-hoc runner
        with su.io.cd(run_root):
            # compile
            su.bash.run(f"javac -cp {proj_info['classpath']} {out_path}", 0)

            # run
            run_cname = pck_rel_path.replace("/", ".") + ".teco_" + cname
            run_cmds = []
            if "junit-4" in proj_info["classpath"]:
                # JUnit 4: first, try to use JUnit runner
                run_cmds.append(
                    f"java -cp .:{proj_info['classpath']} org.junit.runner.JUnitCore {run_cname}"
                )
            # use the ad-hoc runner
            run_cmds.append(f"java -cp .:{proj_info['classpath']} {run_cname}")

            for run_cmd in run_cmds:
                rr_test = su.bash.run(
                    run_cmd, timeout=self.config["timeout_per_test"], warn_nonzero=False
                )
                if rr_test.returncode == 0:
                    break
                # else:
                #     logger.warning(
                #         f"{run_cmd} failed, return code: {rr_test.returncode}\nstdout: {rr_test.stdout}\nstderr: {rr_test.stderr}\n"
                #     )

            # make sure we had a successful run
            if rr_test.returncode != 0:
                raise RuntimeError(
                    f"test failed with return code: {rr_test.returncode}\nstdout: {rr_test.stdout}\nstderr: {rr_test.stderr}\n"
                )

        # 4. collect results
        # print(f"- sign: {' '.join(data.test_sign.get_tokens())}")
        data.runtime_values_depth1 = []
        for stmt_i in range(len(data.test_stmts) + 1):
            this_runtime_values = {}

            primitive_values_file = log_path / f"primitive-values-{stmt_i}"
            if primitive_values_file.exists():
                for line in su.io.load(primitive_values_file, su.io.Fmt.txtList):
                    name, value = line.split(" ", 1)
                    if len(value) > self.config["max_value_char"]:
                        value = value[: self.config["max_value_char"]]
                        self.log_to_file(
                            f"WARNING: data #{data.id} {data.test_mkey} stmt#{stmt_i}, truncating a value of length {len(value)} to {self.config['max_value_char']}"
                        )
                    this_runtime_values[name] = value

            depth1_values_file = log_path / f"depth1-values-{stmt_i}"
            if depth1_values_file.exists():
                for line in su.io.load(depth1_values_file, su.io.Fmt.txtList):
                    name, value = line.split(" ", 1)
                    if len(value) > self.config["max_value_char"]:
                        value = value[: self.config["max_value_char"]]
                        self.log_to_file(
                            f"WARNING: data #{data.id} {data.test_mkey} stmt#{stmt_i}, truncating a value of length {len(value)} to {self.config['max_value_char']}"
                        )
                    this_runtime_values[name] = value

            data.runtime_values_depth1.append(this_runtime_values)

            # print(f"- stmt#{stmt_i}")
            # print(f"  - runtime_types: {this_runtime_types}")
            # print(f"  - runtime_values: {this_runtime_values}")
            # if stmt_i < len(data.test_stmts):
            #     print(f"  - stmt: {' '.join(data.test_stmts[stmt_i].get_tokens())}")

        su.io.rm(out_path)
        su.io.rmdir(temp_dir)

    def filter_nontrivial_values(self, name: str, data_dir: su.arg.RPath):
        if name == "runtime_values":
            target = "runtime_values_nontrivial"
        elif name == "runtime_values_depth1":
            target = "runtime_values_depth1_nontrivial"
        else:
            raise RuntimeError(f"unknown data: {name}")

        # load dataset
        dataset = load_dataset(data_dir, clz=Data, only=["test_stmts", name])

        for data in tqdm(dataset, desc="filtering"):
            src_dicts = getattr(data, name)
            target_dicts = []
            if src_dicts is None or len(src_dicts) != len(data.test_stmts) + 1:
                continue

            constants = set()
            for stmt_i in range(len(data.test_stmts) + 1):
                src_dict = src_dicts[stmt_i]
                target_dict = {}

                # keep non-constant values
                for key, value in src_dict.items():
                    if value not in constants:
                        target_dict[key] = value

                if stmt_i < len(data.test_stmts):
                    # find the constants
                    for node in data.test_stmts[stmt_i].traverse():
                        if node.tok_kind == Consts.tok_literal:
                            constants.add(node.tok)

                target_dicts.append(target_dict)

            setattr(data, target, target_dicts)

        # save the filtered fields
        save_dataset(data_dir, dataset, only=[target])


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(ExtendRuntimeDataCollector, as_positional=False)
