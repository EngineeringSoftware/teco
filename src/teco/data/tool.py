import subprocess
from pathlib import Path
from typing import Optional

import seutil as su

from teco.macros import Macros

EXPECTED_MAVEN_VERSION = "3.8"
EXPECTED_JAVA_VERSION = "1.8.0"


def ensure_tool_versions():
    maven_version = su.bash.run(
        r"mvn --version | grep 'Apache Maven' | sed -nE 's/Apache Maven ([0-9.]+).*/\1/p'"
    ).stdout.strip()
    if EXPECTED_MAVEN_VERSION not in maven_version:
        raise RuntimeError(
            f"Expected Maven version {EXPECTED_MAVEN_VERSION}, but got {maven_version}"
        )
    java_version = su.bash.run(
        r"mvn --version | grep 'Java version:' | sed -nE 's/Java version: ([0-9.]+).*/\1/p'"
    ).stdout.strip()
    if EXPECTED_JAVA_VERSION not in java_version:
        raise RuntimeError(
            f"Expected Java version {EXPECTED_JAVA_VERSION}, but got {java_version}"
        )


class DataCollector:

    data_collector_dir: Path = Macros.project_dir / "data-collector"
    data_collector_version: str = "0.1-dev"

    static_name = "static-collector"
    static_jar: str = str(
        data_collector_dir
        / static_name
        / "target"
        / f"{static_name}-{data_collector_version}-jar-with-dependencies.jar"
    )

    bcverifier_name = "bcverifier"
    bcverifier_jar: str = str(
        data_collector_dir
        / bcverifier_name
        / "target"
        / f"{bcverifier_name}-{data_collector_version}-jar-with-dependencies.jar"
    )

    adhoc_runner_name = "adhoc-runner"
    adhoc_runner_jar: str = str(
        data_collector_dir
        / adhoc_runner_name
        / "target"
        / f"{adhoc_runner_name}-{data_collector_version}-jar-with-dependencies.jar"
    )

    compiled = False

    @classmethod
    def require_compiled(cls):
        if not cls.compiled:
            with su.io.cd(cls.data_collector_dir):
                su.bash.run(f"mvn package -DskipTests", 0)
                cls.compiled = True

    @classmethod
    def run_static(
        cls,
        main: str,
        config: Optional[dict] = None,
        args: Optional[str] = None,
        timeout: Optional[int] = None,
        check_returncode: int = 0,
        jvm_args: str = "",
    ) -> subprocess.CompletedProcess:
        cls.require_compiled()

        if config is not None and args is not None:
            raise ValueError("Cannot specify both config and args")

        if config is not None:
            config_file = su.io.mktmp("dcstatic", ".json")
            su.io.dump(config_file, config)
            args = config_file

        if args is None:
            args = ""

        rr = su.bash.run(
            f"java {jvm_args} -cp {cls.static_jar} {main} {args}",
            check_returncode=check_returncode,
            timeout=timeout,
        )

        if config is not None:
            # delete temp input file
            su.io.rm(config_file)

        return rr
