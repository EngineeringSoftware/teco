import dataclasses
import enum
import re
from typing import List, Optional

import seutil as su

from teco.data.data import Data
from teco.data.structures import Consts
from teco.model.subtokenizer import Subtokenizer

logger = su.log.get_logger(__name__)


def reverse_insns_toks(toks: List[str]) -> List[str]:
    insns = []
    for tok in toks:
        if tok in Consts.ops_all:
            insns.append([tok])
        else:
            if len(insns) == 0:
                # shouldn't start with a non-op token, but we tolerate it
                insns.append([])
            insns[-1].append(tok)
    return sum(reversed(insns), [])


class Input(enum.Enum):
    input_name = (
        "input_name"  # used as the first item, to insert input names before all inputs
    )
    focalm = "focalm"
    sign = "sign"
    prev_stmts = "prev_stmts"
    prev_insns = "prev_insns"
    cur_insn = "cur_insn"
    types_local = "types_local"
    types_absent = "types_absent"
    fields_set = "fields_set"
    fields_notset = "fields_notset"
    setup_teardown = "setup_teardown"
    last_called_method = "last_called_method"
    similar_stmt_all = "similar_stmt_all"
    similar_stmt_1_0 = "similar_stmt_1_0"
    similar_stmt_2_5 = "similar_stmt_2.5"
    runtime_types_values_delta = "runtime_types_values_delta"
    runtime_types_values_all = "runtime_types_values_all"


class Output(enum.Enum):
    stmt = "stmt"
    insn = "insn"
    revinsn = "revinsn"


@dataclasses.dataclass
class InputSequence:
    """
    An input sequence tokens + subtoken ids, with bos/eos tokens.
    """

    toks: List[str] = dataclasses.field(default_factory=list)
    tok_stids: List[List[int]] = dataclasses.field(default_factory=list)
    prefix_of: List[Optional[Input]] = dataclasses.field(default_factory=list)

    def get_stids(self) -> List[int]:
        return sum(self.tok_stids, [])

    def append_tok(
        self, tok: str, subtokenizer: Subtokenizer, prefix_of: Optional[Input] = None
    ) -> None:
        self.toks.append(tok)
        self.tok_stids.append(subtokenizer.toks2stids([tok]))
        self.prefix_of.append(prefix_of)

    def __len__(self):
        return len(self.toks)


def _collect_toks(
    seq: InputSequence,
    toks: List[str],
    subtokenizer: Subtokenizer,
    prefix_of: Optional[Input] = None,
):
    for t in toks:
        seq.append_tok(t, subtokenizer, prefix_of)


def get_input_sequence(
    data: Data,
    stmt_i: int,
    inputs: List[Input],
    subtokenizer: Subtokenizer,
    break_down_desc: bool = True,
    first_sep: bool = False,
):
    seq = InputSequence()
    _collect_toks(seq, [subtokenizer.bos_token], subtokenizer)
    use_input_name = False

    for input_i, input_ in enumerate(inputs):
        if input_i != 0 or first_sep:
            _collect_toks(seq, [subtokenizer.sep_token], subtokenizer, prefix_of=input_)
        if use_input_name:
            _collect_toks(seq, input_.name, subtokenizer, prefix_of=input_)

        if input_ == Input.input_name:
            use_input_name = True
            continue
        elif input_ == Input.focalm:
            _collect_toks(seq, data.focalm.get_tokens(), subtokenizer)
        elif input_ == Input.sign:
            _collect_toks(seq, data.test_sign.get_tokens(), subtokenizer)
        elif input_ == Input.prev_stmts:
            for i in range(stmt_i):
                _collect_toks(seq, data.test_stmt_toks[i], subtokenizer)
        elif input_ == Input.prev_insns:
            toks = []
            for i in range(stmt_i):
                toks += data.test_stmt_insns[i]
            if break_down_desc:
                toks = preprocess_break_down_desc(toks)
            _collect_toks(seq, toks, subtokenizer)
        elif input_ == Input.cur_insn:
            toks = data.test_stmt_insns[stmt_i]
            if break_down_desc:
                toks = preprocess_break_down_desc(toks)
            _collect_toks(seq, toks, subtokenizer)
        elif input_ == Input.types_local:
            _collect_toks(seq, data.types_local_simplified[stmt_i], subtokenizer)
        elif input_ == Input.types_absent:
            _collect_toks(seq, data.types_absent_simplified[stmt_i], subtokenizer)
        elif input_ == Input.fields_set:
            _collect_toks(seq, data.fields_set[stmt_i], subtokenizer)
        elif input_ == Input.fields_notset:
            _collect_toks(seq, data.fields_notset[stmt_i], subtokenizer)
        elif input_ == Input.setup_teardown:
            for m in data.setup_methods:
                _collect_toks(seq, m.get_tokens(), subtokenizer)
            for m in data.teardown_methods:
                _collect_toks(seq, m.get_tokens(), subtokenizer)
        elif input_ == Input.last_called_method:
            lcm = data.resolve_last_called_method(stmt_i)
            if lcm is not None:
                _collect_toks(seq, lcm.get_tokens(), subtokenizer)
        elif input_ in {
            Input.similar_stmt_all,
            Input.similar_stmt_1_0,
            Input.similar_stmt_2_5,
        }:
            if data.similar_stmts is not None:
                similar_stmt = data.similar_stmts[stmt_i]
                if similar_stmt is not None:
                    score, stmt = similar_stmt
                    if input_ == Input.similar_stmt_1_0 and score < 1.0:
                        continue
                    elif input_ == Input.similar_stmt_2_5 and score < 2.5:
                        continue
                    _collect_toks(seq, stmt.get_tokens(), subtokenizer)
        elif input_ == Input.runtime_types_values_delta:
            if data.runtime_data_valid():
                toks = []
                for k, (t, v) in sorted(
                    data.runtime_types_values_delta[stmt_i].items()
                ):
                    toks.append(k)
                    toks.append("=")
                    toks.append("(")
                    toks.append(t)
                    toks.append(")")
                    toks.append(v)
                    toks.append(";")
                _collect_toks(seq, toks, subtokenizer)
        elif input_ == Input.runtime_types_values_all:
            if data.runtime_data_valid():
                toks = []
                for k, (t, v) in sorted(
                    data.runtime_types_values_all[stmt_i].items()
                ):
                    toks.append(k)
                    toks.append("=")
                    toks.append("(")
                    toks.append(t)
                    toks.append(")")
                    toks.append(v)
                    toks.append(";")
                _collect_toks(seq, toks, subtokenizer)
        else:
            raise RuntimeError(f"unknown input {input_}")

    _collect_toks(seq, [subtokenizer.eos_token], subtokenizer)
    return seq


def get_output_ids(
    data: Data,
    stmt_i: int,
    output: Output,
    subtokenizer: Subtokenizer,
    break_down_desc: bool = True,
) -> List[int]:
    if output == Output.stmt:
        toks = data.test_stmt_toks[stmt_i]
    elif output == Output.insn:
        toks = data.test_stmt_insns[stmt_i]
    elif output == Output.revinsn:
        toks = reverse_insns_toks(data.test_stmt_insns[stmt_i])
    else:
        raise ValueError(f"Unsupported output type: {output}")

    if break_down_desc:
        toks = preprocess_break_down_desc(toks)

    return (
        [subtokenizer.bos_token_id]
        + subtokenizer.toks2stids(toks)
        + [subtokenizer.eos_token_id]
    )


re_class_desc = re.compile(r"\[+[BCDFIJSZ]|\[*L[a-zA-Z_$][a-zA-Z_$0-9]*;")
re_method_desc = re.compile(
    r"\((\[*([BCDFIJSZ]|L[a-zA-Z_$][a-zA-Z_$0-9]*;))*\)(V|\[*([BCDFIJSZ]|L[a-zA-Z_$][a-zA-Z_$0-9]*;))"
)


def preprocess_break_down_desc(toks: List[str]) -> List[str]:
    ret = []
    for tok in toks:
        if (
            re_class_desc.fullmatch(tok) is not None
            or re_method_desc.fullmatch(tok) is not None
        ):
            in_ref_desc = False
            for c in tok:
                if in_ref_desc:
                    if c == ";":
                        ret.append(c)
                        in_ref_desc = False
                    else:
                        ret[-1] += c
                elif c == "L":
                    ret.append(c)
                    ret.append("")
                    in_ref_desc = True
                else:
                    ret.append(c)
        else:
            ret.append(tok)
    return ret


def postprocess_recover_break_down_desc(toks: List[str]) -> List[str]:
    """
    Should only be called on instructions tokens; may have side effects to other kinds of tokens.
    """
    ret = []
    i = 0
    cur_desc = []
    in_method_desc = False
    while i < len(toks):
        cur_tok = toks[i]
        if cur_tok == "[":
            cur_desc.append(cur_tok)
        elif cur_tok == "L" and i + 2 < len(toks) and toks[i + 2] == ";":
            cur_desc.append(cur_tok)
            cur_desc.append(toks[i + 1])
            cur_desc.append(toks[i + 2])
            if not in_method_desc:
                # commit the current desc (either a class desc or the return type of a method desc)
                ret.append("".join(cur_desc))
                cur_desc = []
            i += 3
            continue
        elif cur_tok == "(":
            in_method_desc = True
            cur_desc.append(cur_tok)
        elif cur_tok == ")" and in_method_desc:
            in_method_desc = False
            cur_desc.append(cur_tok)
            if i + 1 < len(toks) and toks[i + 1] in set("BCDFIJSZV"):
                # consume the return type right now
                cur_desc.append(toks[i + 1])
                ret.append("".join(cur_desc))
                cur_desc = []
                i += 2
                continue
        elif cur_tok in set("BCDFIJSZ") and in_method_desc:
            cur_desc.append(cur_tok)
        elif cur_tok in set("BCDFIJSZ") and len(cur_desc) > 0 and cur_desc[-1] == "[":
            cur_desc.append(cur_tok)
            ret.append("".join(cur_desc))
            cur_desc = []
        else:
            if len(cur_desc) > 0:
                # should not happen, but we tolerate it
                # logger.warning(
                #     f"current desc unfinished but got a wierd token: {cur_desc=}, {in_method_desc=}, {cur_tok=}"
                # )
                ret += cur_desc
                cur_desc = []
                in_method_desc = False
            ret.append(cur_tok)
        i += 1
    if len(cur_desc) > 0:
        # should not happen, but we tolerate it
        # logger.warning(
        #     f"current desc unfinished but got EOF: {cur_desc=}, {in_method_desc=}"
        # )
        ret += cur_desc
    return ret


def subtoks2toks(subtoks: List[str]) -> List[str]:
    toks = "".join(subtoks).split(" ")
    if len(toks) > 0 and toks[0] == "":
        toks = toks[1:]
    return toks
