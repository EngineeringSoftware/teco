from typing import List, Optional

import seutil as su

from teco.exe.constraints import InsnConstraint
from teco.exe.ng_wrapper import NailgunProgram

logger = su.log.get_logger(__name__)


class BytecodeConstrainer(NailgunProgram):
    def __init__(self):
        super().__init__(main_class="org.teco.BytecodeConstrainer")

    def start_session(self, args: Optional[List[str]] = None) -> int:
        if args is None:
            args = []
        return self.call(["start_session"] + args, deserialize=int)

    def fork_session(self, session_id: int) -> int:
        return self.call(["fork_session", str(session_id)], deserialize=int)

    def end_session(self, session_id: int):
        self.call(["end_session", str(session_id)])

    def try_step(self, session_id: int, toks: List[str]) -> InsnConstraint:
        return self.call(
            ["try_step", str(session_id)] + toks, deserialize_type=InsnConstraint
        )

    def submit_step(self, session_id: int):
        return self.call(["submit_step", str(session_id)])
