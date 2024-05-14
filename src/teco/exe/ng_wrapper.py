import io
import json
import os
import subprocess
from typing import Any, Callable, List, Optional, Union, Type

import seutil as su

from teco.data.tool import DataCollector, ensure_tool_versions
from teco.exe import ng

logger = su.log.get_logger(__name__, level=su.log.INFO)


class NailgunProgram:
    """
    Wrapper for a Java program that we connect to via nailgun.
    """

    def __init__(self, main_class: str, jar: str = DataCollector.bcverifier_jar):
        self.main_class = main_class
        self.jar = jar

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, type, value, tb):
        self.teardown()

    def setup(self):
        ensure_tool_versions()
        DataCollector.require_compiled()

        # start nailgun server
        logger.info("starting nailgun server")
        self.tmpdir = su.io.mktmp_dir("teco-nailgun")
        self.transport_file = self.tmpdir / "sock"
        self.transport_address = f"local:{self.transport_file}"
        self.heartbeat_timeout_ms = 10_000

        def preexec_fn():
            # Close any open file descriptors to further separate buckd from its
            # invoking context (e.g. otherwise we'd hang when running things like
            # `ssh localhost buck clean`).
            dev_null_fd = os.open("/dev/null", os.O_RDWR)
            os.dup2(dev_null_fd, 0)
            os.dup2(dev_null_fd, 2)
            os.close(dev_null_fd)

        cmd = [
            "java",
            "-Djna.nosys=true",
            "-classpath",
            self.jar,
            "com.facebook.nailgun.NGServer",
            self.transport_address,
            str(self.heartbeat_timeout_ms),
        ]

        self.ng_server_process = subprocess.Popen(
            cmd, preexec_fn=preexec_fn, stdout=subprocess.PIPE
        )

        # wait for the server to be ready
        while True:
            the_first_line = str(self.ng_server_process.stdout.readline().strip())
            if "NGServer" in the_first_line and "started" in the_first_line:
                break
            if the_first_line is None or the_first_line == "":
                break
        logger.info("nailgun server is ready")

    def teardown(self):
        # stop nailgun server
        logger.info("stopping nailgun server")
        try:
            with ng.NailgunConnection(
                self.transport_address,
                cwd=os.getcwd(),
                stderr=None,
                stdin=None,
                stdout=None,
            ) as c:
                c.send_command("ng-stop")
        except ng.NailgunException as e:
            logger.warning(f"Failed to stop nailgun server safely: {e}")

        process_exit_code = self.ng_server_process.wait(1)
        if process_exit_code is None:
            self.ng_server_process.kill()
        su.io.rmdir(self.tmpdir)

    def call(
        self,
        args: List[str],
        expected_exit_code: Optional[int] = 0,
        warn_stderr: bool = True,
        deserialize: Optional[Callable[[str], Any]] = None,
        deserialize_type: Type = None,
    ) -> Union[str, Any]:
        """
        Invokes the main class's main method with the given arguments.

        The input arguments are passed as a list of strings.
        The return value is obtained by reading the stdout of the Java program, as a string.
        (As a result, be careful not to add any System.out.print that will mess up the return value.)

        :param args: the arguments to pass to the main method.
        :param expected_exit_code: the expected exit code of the Java program (by default 0); if set to None, the exit code is not checked.
        :param warn_stderr: if True, the stderr of the Java program is warned via the logger.
        :param deserialize: if provided, used to deserialized the return value.
        :param deserialize_type: if provided, assume the return value string is json and try to deserialize it to the specified type with su.io.deserialize.
        :return: the stdout of the Java program.
        """
        if deserialize is not None and deserialize_type is not None:
            raise RuntimeError("deserialize and deserialize_type cannot be both provided")

        stdout = io.BytesIO()
        stderr = io.BytesIO()
        with ng.NailgunConnection(
            self.transport_address, stdout=stdout, stderr=stderr, stdin=None
        ) as c:
            exit_code = c.send_command(self.main_class, args)

        stdout_txt = stdout.getvalue().decode("utf8")
        stderr_txt = stderr.getvalue().decode("utf8")

        if expected_exit_code is not None:
            if exit_code != expected_exit_code:
                raise RuntimeError(
                    f"Expected exit code {expected_exit_code}, but got {exit_code}, when running {self.main_class} {args};\n stdout: {stdout_txt}\n stderr: {stderr_txt}"
                )

        if warn_stderr:
            if len(stderr_txt) != 0:
                logger.warning(f"stderr: {stderr_txt}")

        if deserialize_type is not None:
            deserialize = lambda x: su.io.deserialize(json.loads(x), clz=deserialize_type)

        if deserialize is not None:
            try:
                return deserialize(stdout_txt)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Failed to deserialize the return value of {self.main_class} {args}: {e}\n stdout: {stdout_txt}\n stderr: {stderr_txt}\n"
                )
        else:
            return stdout_txt
