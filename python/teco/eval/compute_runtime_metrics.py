import collections
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import TimeoutExpired
from typing import List, Tuple

import seutil as su
from jsonargparse import CLI
from seutil.project import Project
from tqdm import tqdm

from teco.data.data import Data
from teco.data.maven import MavenProjectHelper
from teco.data.tool import DataCollector, ensure_tool_versions
from teco.data.utils import load_dataset, save_dataset
from teco.macros import Macros
from teco.model.prediction import Prediction
from teco.utils import aggregate_metrics, summarize_metrics

logger = su.log.get_logger(__name__)


class RuntimeMetricsComputer:

    CLONE_TIMEOUT = 300
    COMPILE_TIMEOUT = 300
    COLLECT_TIMEOUT = 100
    BATCH_TIMEOUT = 43200  # 12h
    PLACE_HOLDER = "/*<teco-place-holder-for-prediction>*/"

    def __init__(
        self,
        downloads_dir: su.arg.RPath = Macros.downloads_dir,
        timeout_per_test: int = 5,
        mode: str = "top1",
        debug: bool = False,
    ):
        assert mode in ["top1", "first-runnable"]
        self.downloads_dir = downloads_dir
        self.timeout_per_test = timeout_per_test
        self.mode = mode
        self.debug = debug

        # check tool versions
        ensure_tool_versions()

    def pack_used_projs(self, data_dir: su.arg.RPath):
        # load dataset
        dataset = load_dataset(
            data_dir,
            clz=Data,
            only=["test_sign", "test_stmts", "proj_name", "test_mkey"],
        )

        # group data by project
        pname2indices = collections.defaultdict(list)
        for i, data in enumerate(dataset):
            pname2indices[data.proj_name].append(i)

        used_pnames = list(pname2indices.keys())

        # print command to pack all used projects
        print(f"# command")
        print(f"  cd {self.downloads_dir}")
        print(f"  tar cf used-downloads.tar {' '.join(used_pnames)}")

    def compute_metrics_batch(
        self,
        pred_dir_list: List[su.arg.RPath],
        data_dir: su.arg.RPath,
        repos_file: su.arg.RPath = Macros.work_dir / "repos" / "filtered" / "used.json",
        batch_size: int = 1_000,
        pool_size: int = 16,
        compile_only: bool = False,
        no_compile: bool = False,
    ):
        if compile_only and no_compile:
            raise ValueError("compile_only and no_compile cannot be both True")

        temp_dir = su.io.mktmp_dir("teco")

        # load dataset
        dataset = load_dataset(
            data_dir,
            clz=Data,
            only=["test_sign", "test_stmts", "proj_name", "test_mkey"],
        )

        # group data by project
        pname2indices = collections.defaultdict(list)
        for i, data in enumerate(dataset):
            pname2indices[data.proj_name].append(i)

        if self.debug:
            logger.warning("debug mode: 10 data per project, and only 10 projects")
            logger.warning(f"temp dir: {temp_dir}")
            for pname in pname2indices:
                pname2indices[pname] = pname2indices[pname][:10]
            pname2indices = dict(list(pname2indices.items())[:10])

        # load all predictions
        preds_list: List[List[Prediction]] = []
        for pred_dir in tqdm(pred_dir_list, desc="loading predictions"):
            preds = su.io.load(pred_dir / "preds.jsonl", clz=Prediction)
            assert len(preds) == len(
                dataset
            ), f"{len(preds)=} != {len(dataset)=}, in {pred_dir}"

            # default values for our new metrics
            for pred in preds:
                pred.metrics["compilable"] = 0
                pred.metrics["runnable"] = 0

            preds_list.append(preds)

        if not no_compile:
            # make sure the data collector is compiled in its latest version
            DataCollector.require_compiled()
        else:
            # assume data collector is already compiled
            DataCollector.compiled = True

        # load and prepare projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        indexed_projects = {p.full_name: p for p in projects}
        used_projects = [indexed_projects[pn] for pn in pname2indices]

        proj2info = {}
        failed_projects = []
        for project in tqdm(used_projects, desc="preparing projects"):
            try:
                proj2info[project.full_name] = self.prepare_project(project, no_compile)
            except KeyboardInterrupt:
                input(
                    "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                )
                logger.warning(
                    f"Processing failed for {project.full_name}: User interrupted"
                )
                failed_projects.append(project.full_name)
                continue
            except:
                logger.warning(
                    f"Processing failed for {project.full_name}: {traceback.format_exc()}"
                )
                failed_projects.append(project.full_name)
                continue

        if compile_only:
            print(f"Finished compiling all projects, {len(failed_projects)} failed")
            return

        # divide into batches and run with thread pool executor
        executor = ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="teco-eval"
        )

        pbar = tqdm(total=len(dataset))
        progress_failed_projects = 0
        for pname in failed_projects:
            progress_failed_projects += len(pname2indices[pname])

        jobs = []
        for pname, proj_info in proj2info.items():
            project = indexed_projects[pname]
            indices = pname2indices[pname]

            beg = 0
            batch_i = 0
            while beg < len(indices):
                indices_batch = indices[beg : beg + batch_size]

                batch_work_dir = temp_dir / pname / f"batch-{batch_i}"

                # dump all the information needed for this project
                su.io.dump(batch_work_dir / "indices.json", indices_batch)
                su.io.dump(batch_work_dir / "proj_info.pkl", proj_info)
                dataset_batch = [dataset[i] for i in indices_batch]
                save_dataset(batch_work_dir / "dataset", dataset_batch)

                predicted_stmts_list: List[List[List]] = []  # [indice, model, topk]
                for i in indices_batch:
                    models_predicted_stmts = []
                    for preds in preds_list:
                        models_predicted_stmts.append(
                            [" ".join(seq["toks"]) for seq in preds[i].topk]
                        )
                    predicted_stmts_list.append(models_predicted_stmts)
                su.io.dump(
                    batch_work_dir / "predicted-stmts-list.pkl", predicted_stmts_list
                )
                jobs.append(
                    (
                        batch_work_dir,
                        f"python -m teco.eval.compute_runtime_metrics --timeout_per_test {self.timeout_per_test} --mode {self.mode} --downloads_dir {self.downloads_dir} --debug {self.debug} compute_metrics_proj --work_dir {batch_work_dir}",
                    )
                )

                batch_i += 1
                beg += batch_size

        # random shuffle the jobs to avoid compiling the same project together (esp. when many parallel jobs are ongoing)
        # to completely avoid potential conflicts caused during compiling projects, perform compilation in a separate step and then run the evaluation with --no-compile
        random.shuffle(jobs)
        work_dirs = []
        futures = []
        for work_dir, job in jobs:
            work_dirs.append(work_dir)
            futures.append(
                executor.submit(
                    lambda job=job: su.bash.run(job, 0, timeout=self.BATCH_TIMEOUT)
                )
            )

        # periodically check for progress
        progress_finished = 0
        fail_finished = 0

        while len(futures) > 0:
            progress_running = 0
            fail_running = 0

            just_finished = []
            for job_i, (work_dir, future) in enumerate(zip(work_dirs, futures)):
                if future.done():
                    indices_batch = su.io.load(work_dir / "indices.json")

                    try:
                        future.result()
                    except:
                        # this batch failed
                        logger.warning(
                            f"Batch failed: {work_dir}; exception: {traceback.format_exc()}"
                        )
                        fail_finished += len(indices_batch)
                    else:
                        # collect results for this batch
                        results_list = su.io.load(work_dir / "results-list.pkl")
                        # print(f"{indices_batch=}")
                        # print(f"{results_list=}")

                        for i, indice in enumerate(indices_batch):
                            models_preds = [preds[indice] for preds in preds_list]
                            models_results_list = results_list[i]
                            if models_results_list is None:
                                # something failed for this indice...
                                models_results_list = [None for _ in models_preds]
                            for results, pred in zip(models_results_list, models_preds):
                                # put per-seq information to misc
                                pred.misc["topk_compilable"] = []
                                pred.misc["topk_runnable"] = []
                                pred.misc["topk_runtime"] = []
                                if results is None:
                                    pred.misc["topk_compilable"].append(0)
                                    pred.misc["topk_runnable"].append(0)
                                    pred.misc["topk_runtime"].append(None)
                                else:
                                    for compilable, runnable, runtime in results:
                                        pred.misc["topk_compilable"].append(compilable)
                                        pred.misc["topk_runnable"].append(runnable)
                                        pred.misc["topk_runtime"].append(runtime)

                                # put top-1 result in metrics
                                pred.metrics["compilable"] = pred.misc[
                                    "topk_compilable"
                                ][0]
                                pred.metrics["runnable"] = pred.misc["topk_runnable"][0]

                        # remove this batch from the list of futures
                        _, fail = su.io.load(work_dir / "progress.json")
                        fail_finished += fail
                    finally:
                        just_finished.append(job_i)
                        progress_finished += len(indices_batch)
                else:
                    # collect current progress
                    if (work_dir / "progress.json").exists():
                        try:
                            total, fail = su.io.load(work_dir / "progress.json")
                        except:
                            # tolerate corrupted progress.json
                            total, fail = 0, 0
                        progress_running += total
                        fail_running += fail

            for job_i in reversed(just_finished):
                work_dirs.pop(job_i)
                futures.pop(job_i)

            progress = progress_finished + progress_running + progress_failed_projects
            fail = fail_finished + fail_running

            pbar.update(progress - pbar.n)
            pbar.set_description(f"remaining batches: {len(futures)}; failed: {fail}")
            time.sleep(10)

        executor.shutdown()
        pbar.close()

        if len(failed_projects) > 0:
            logger.warning(
                f"In total {len(failed_projects)} projects failed: {failed_projects}"
            )
        if fail_finished > 0:
            logger.warning(f"In total {fail_finished} data failed")

        if self.debug:
            print(f"debugging, stop here")
            raise KeyboardInterrupt

        # save the predictions
        for pred_dir, preds in tqdm(
            zip(pred_dir_list, preds_list),
            desc="saving updated predictions",
            total=len(preds_list),
        ):
            su.io.dump(pred_dir / "preds.jsonl", preds)

            # recompute summary metrics
            metrics = aggregate_metrics([pred.metrics for pred in preds])
            metrics_summary = summarize_metrics(metrics)
            su.io.dump(
                pred_dir / "metrics_summary.json", metrics_summary, su.io.Fmt.jsonNoSort
            )

        su.io.rmdir(temp_dir)

    def compute_metrics_proj(self, work_dir: su.arg.RPath):
        proj_info = su.io.load(work_dir / "proj_info.pkl")
        dataset = load_dataset(work_dir / "dataset", clz=Data)
        predicted_stmts_list = su.io.load(work_dir / "predicted-stmts-list.pkl")
        su.io.mkdir(work_dir / "tmp")

        # prepare the adhoc runner generator
        # self.generator = AdHocRunnerGenerator("org.teco.AdHocRunnerGeneratorEval")
        # self.generator.setup()
        # atexit.register(self.generator.teardown)

        total = 0
        fail = 0
        results_list = []
        for data, models_predicted_stmts in zip(dataset, predicted_stmts_list):
            models_results = None
            try:
                models_results = self.eval_data(
                    data, proj_info, models_predicted_stmts, work_dir
                )
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(
                    f"data #{data.id} {data.test_mkey} stmt {len(data.test_stmts)} failed: {traceback.format_exc()}"
                )
                fail += 1
                continue
            finally:
                total += 1
                results_list.append(models_results)
                su.io.dump(work_dir / "progress.json", (total, fail))

        # save results
        su.io.dump(work_dir / "results-list.pkl", results_list)

    def prepare_project(self, proj: Project, no_compile: bool = False) -> dict:
        if no_compile:
            # assume the project is already cloned and compiled properly
            proj.set_cloned_dir(self.downloads_dir / proj.full_name)
        else:
            # clone the project
            with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
                proj.clone(self.downloads_dir)
            proj.checkout(proj.data["sha"], forced=True)

            # compile all classes (including application and test classes)
            with su.TimeUtils.time_limit(self.COMPILE_TIMEOUT):
                with su.io.cd(proj.dir):
                    rr = su.bash.run(
                        f"mvn test-compile {Macros.MVN_SKIPS}",
                        timeout=self.COMPILE_TIMEOUT,
                    )
                    if rr.returncode != 0:
                        # try to clean compile once again
                        su.bash.run(
                            f"mvn clean test-compile {Macros.MVN_SKIPS}",
                            0,
                            timeout=self.COMPILE_TIMEOUT,
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

    def eval_data(
        self,
        data: Data,
        proj_info: dict,
        models_predicted_stmts: List[List[str]],
        work_dir: Path,
    ) -> List[List[Tuple[int, int, float]]]:
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

        # 2. modify this file to get an ad-hoc runner for the test with prediction place holder
        # prepare a temp dir
        temp_dir = su.io.mktmp_dir(prefix=data.proj_name, dir=work_dir / "tmp")

        # generate the modified test at the temp directory with same package structure (with teco_ prefix)
        # test resources should be available on classpath, except for some cornel cases
        run_root = temp_dir / "java"
        pck_rel_path = "/".join(data.test_mkey.split("#")[0].split("/")[:-1])
        out_path = run_root / pck_rel_path / f"teco_{cname}.java.template"
        su.io.mkdir(out_path.parent)

        # call the generator (in Java)
        su.bash.run(
            f"java -cp {DataCollector.adhoc_runner_jar} org.teco.AdHocRunnerGeneratorEval {str(src_path.absolute())} {cname} {mname} {str(out_path.absolute())} {len(data.test_stmts)}",
            0,
        )
        template = su.io.load(out_path, su.io.Fmt.txt)

        models_results = []
        with su.io.cd(run_root):
            for model_i, predicted_stmts in enumerate(models_predicted_stmts):
                results = []
                for seq_i, stmt in enumerate(predicted_stmts):
                    # for each seq in each prediction
                    compilable = 0
                    runnable = 0
                    runtime = None

                    if len(stmt.strip()) != 0:
                        # 3. generate the actual runner with place holder replaced to the predicted sequence
                        src = template.replace(self.PLACE_HOLDER, stmt)
                        src_path = run_root / pck_rel_path / f"teco_{cname}.java"
                        su.io.dump(src_path, src, su.io.Fmt.txt)

                        # 4. compile and run the ad-hoc runner
                        rr_compile = su.bash.run(
                            f"javac -cp {proj_info['classpath']} {src_path}"
                        )
                        if rr_compile.returncode != 0:
                            compilable = 0
                            runnable = 0
                            runtime = None
                        else:
                            compilable = 100

                            # run
                            run_cname = (
                                pck_rel_path.replace("/", ".") + ".teco_" + cname
                            )
                            run_cmds = []
                            if "junit-4" in proj_info["classpath"]:
                                # JUnit 4: first, try to use JUnit runner
                                run_cmds.append(
                                    f"java -cp .:{proj_info['classpath']} org.junit.runner.JUnitCore {run_cname}"
                                )
                            # use the ad-hoc runner
                            run_cmds.append(
                                f"java -cp .:{proj_info['classpath']} {run_cname}"
                            )

                            test_success = False
                            for run_cmd in run_cmds:
                                time_beg = time.time()
                                try:
                                    with su.TimeUtils.time_limit(self.timeout_per_test):
                                        rr_test = su.bash.run(
                                            run_cmd,
                                            timeout=self.timeout_per_test,
                                            warn_nonzero=False,
                                        )
                                except (su.TimeoutException, TimeoutExpired):
                                    continue
                                time_end = time.time()
                                last_runtime = time_end - time_beg
                                if rr_test.returncode == 0:
                                    test_success = True

                            # make sure we had a successful run
                            if not test_success:
                                runnable = 0
                                runtime = None
                            else:
                                runnable = 100
                                runtime = last_runtime

                    results.append((compilable, runnable, runtime))
                    if self.debug:
                        su.io.dump(
                            work_dir / "debug.txt",
                            [
                                f"{data.id}, model#{model_i}, seq#{seq_i}: {compilable} {runnable} {runtime}"
                            ],
                            su.io.Fmt.txtList,
                            append=True,
                        )
                    if self.mode == "first-runnable" and runnable == 100:
                        # stop after first runnable
                        break
                    if self.mode == "top1":
                        # stop after first seq in each prediction
                        break
                models_results.append(results)

        if not self.debug:
            su.io.rmdir(temp_dir)
        return models_results


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(RuntimeMetricsComputer, as_positional=False)
