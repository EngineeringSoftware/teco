from teco.data.tool import DataCollector
from teco.exe.ng_wrapper import NailgunProgram


class AdHocRunnerGenerator(NailgunProgram):
    def __init__(self, main_class: str):
        super().__init__(
            main_class=main_class,
            jar=DataCollector.adhoc_runner_jar,
        )

    def generate(
        self, src_path: str, cname: str, mname: str, out_path: str, log_path: str
    ) -> None:
        self.call(args=["generate", src_path, cname, mname, out_path, log_path])
