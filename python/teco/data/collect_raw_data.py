import atexit
import traceback
from pathlib import Path
from typing import List, Union

import seutil as su
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr
from seutil.project import Project
from tqdm import tqdm

from teco.data.maven import MavenProjectHelper
from teco.data.tool import DataCollector, ensure_tool_versions
from teco.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


class RawDataCollector:
    """
    Class responsible for collecting metrics about Corpus and Generating AST
    Invokes Java collectors to accomplish this
    """

    CLONE_TIMEOUT = 300
    COMPILE_TIMEOUT = 300
    COLLECT_TIMEOUT = 1200
    TEST_TIMEOUT = 1800

    def __init__(
        self,
        downloads_dir: Union[Path_drw, Path_dc, Path] = Macros.downloads_dir,
        debug: bool = False,
    ):
        if not isinstance(downloads_dir, Path):
            downloads_dir = Path(downloads_dir.abs_path)
        self.downloads_dir = downloads_dir

        self.debug = debug

    def collect_raw_data(
        self,
        repos_file: Path_fr,
        out_dir: Union[Path_dc, Path_drw],
        project_names: List[str] = None,
        skip_collected: bool = True,
    ):
        repos_file = Path(repos_file.abs_path)
        self.out_dir = Path(out_dir.abs_path)

        # check tool versions
        ensure_tool_versions()

        su.io.mkdir(self.out_dir)

        # load projects
        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        projects.sort(key=lambda p: p.full_name)

        # limit to user specified projects
        if project_names is not None:
            projects = [p for p in projects if p.full_name in project_names]
            logger.info(
                f"Selected {len(projects)} projects: {[p.full_name for p in projects]}"
            )

        # start collecting
        success = 0
        skip = 0
        fail = 0
        failed = []
        atexit.register(
            lambda: print(
                f"# To clean up failed results:\n  ( cd {out_dir} && rm -rf {' '.join(failed)} )"
                if len(failed) > 0
                else "# No failed projects, cheers!!!"
            )
        )

        pbar = tqdm(desc=f"Collecting jre", total=len(projects))
        jre_out_dir = self.out_dir / "jre"
        if jre_out_dir.exists() and skip_collected:
            logger.info("Skipping jre")
        else:
            with su.TimeUtils.time_limit(self.COLLECT_TIMEOUT):
                self.collect_jre()

        for p in projects:
            pbar.set_description(
                f"Collecting {p.full_name} (+{success} -{fail} s{skip})"
            )

            project_out_dir = self.out_dir / p.full_name
            if project_out_dir.exists() and skip_collected:
                logger.info(f"Skipping {p.full_name}")
                pbar.update(1)
                skip += 1
                continue

            try:
                with su.TimeUtils.time_limit(self.COLLECT_TIMEOUT):
                    self.collect_raw_data_project(p)
                success += 1
            except KeyboardInterrupt:
                input(
                    "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                )
                logger.warning(f"Processing failed for {p.full_name}: User interrupted")
                fail += 1
                failed.append(p.full_name)
            except BaseException:
                logger.warning(
                    f"Processing failed for {p.full_name}: {traceback.format_exc()}"
                )
                fail += 1
                failed.append(p.full_name)
            finally:
                pbar.update(1)
        pbar.set_description(f"Finished (+{success} -{fail} s{skip})")

    def collect_jre(self):
        # prepare output directory
        jre_out_dir = self.out_dir / "jre"
        su.io.mkdir(jre_out_dir, fresh=True)

        # find jre class path
        java_exe = Path(su.bash.run("which java").stdout.strip())
        jre_class_path = java_exe.parent.parent / "jre" / "lib" / "rt.jar"

        # run data collector
        DataCollector.run_static(
            main="org.teco.AllCollectors",
            config={
                "jreClassPath": str(jre_class_path),
                "outputDir": str(jre_out_dir),
                "debug": self.debug,
                "debugPath": str(Macros.debug_dir),
            },
            timeout=self.COLLECT_TIMEOUT,
        )

    def collect_raw_data_project(self, p: Project):
        with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
            p.clone(self.downloads_dir)
        p.checkout(p.data["sha"], forced=True)

        # compile all classes (including application and test classes)
        with su.TimeUtils.time_limit(self.COMPILE_TIMEOUT):
            with su.io.cd(p.dir):
                rr = su.bash.run(f"mvn test-compile", timeout=self.COMPILE_TIMEOUT)
                if rr.returncode != 0:
                    # try to clean compile once again
                    su.bash.run(
                        f"mvn clean test-compile", 0, timeout=self.COMPILE_TIMEOUT
                    )

        # prepare output directory
        project_out_dir = self.out_dir / p.full_name
        su.io.mkdir(project_out_dir, fresh=True)

        # run data collector
        DataCollector.run_static(
            main="org.teco.AllCollectors",
            config={
                "appSrcPath": MavenProjectHelper.get_app_src_path(p),
                "testSrcPath": MavenProjectHelper.get_test_src_path(p),
                "appClassPath": MavenProjectHelper.get_app_class_path(p),
                "testClassPath": MavenProjectHelper.get_test_class_path(p),
                "dependencyClassPath": MavenProjectHelper.get_dependency_classpath(p),
                "jreDataPath": str(self.out_dir / "jre"),
                "outputDir": str(project_out_dir),
                "debug": self.debug,
                "debugPath": str(Macros.debug_dir),
            },
            timeout=self.COLLECT_TIMEOUT,
            jvm_args="-Xss1g",
        )


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(RawDataCollector, as_positional=False)
