from pathlib import Path
import os


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    log_file: Path = python_dir / "experiment.log"
    debug_dir: Path = python_dir / "debug"
    project_dir: Path = python_dir.parent

    results_dir: Path = project_dir / "results"
    work_dir: Path = project_dir / "_work"
    downloads_dir: Path = work_dir / "downloads"

    all_set = "all"
    train = "train"
    val = "val"
    test = "test"

    MVN_SKIPS = "-Djacoco.skip -Dcheckstyle.skip -Drat.skip -Denforcer.skip -Danimal.sniffer.skip -Dmaven.javadoc.skip -Dfindbugs.skip -Dwarbucks.skip -Dmodernizer.skip -Dimpsort.skip -Dpmd.skip -Dxjc.skip -Dair.check.skip-all"
