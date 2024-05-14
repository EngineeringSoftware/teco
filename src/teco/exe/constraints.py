import abc
import copy
import dataclasses
import functools
import re
from typing import Dict, List, Optional, Set, Tuple

from teco.data.structures import FieldStructure, Insn, MethodStructure
from teco.model.processing import (
    postprocess_recover_break_down_desc,
    preprocess_break_down_desc,
)
from teco.utils import Trie


re_cdesc = re.compile(
    r"L(([a-zA-Z_][a-zA-Z_0-9]*)/)+(?P<canonical>[a-zA-Z_$][a-zA-Z_$0-9]*);"
)
re_fqname = re.compile(
    r"(([a-zA-Z_][a-zA-Z_0-9]*)/)+(?P<canonical>[a-zA-Z_$][a-zA-Z_$0-9]*)"
)


@dataclasses.dataclass
class BasicValue:
    """
    Mirror of org.objectweb.asm.tree.analysis.BasicValue
    """

    tname: str = ""

    @classmethod
    def deserialize(cls, tname: str) -> "BasicValue":
        return BasicValue(tname=tname)

    def serialize(self) -> str:
        return self.tname

    def __repr__(self) -> str:
        return self.tname

    def is_uninitialized(self) -> bool:
        return self.tname == "."

    def is_int(self) -> bool:
        return self.tname in {"I", "Z", "B", "S", "C"}

    def is_byte_or_boolean(self) -> bool:
        return self.tname in {"I", "Z", "B"}

    def is_char(self) -> bool:
        return self.tname in {"I", "C"}

    def is_double(self) -> bool:
        return self.tname == "D"

    def is_float(self) -> bool:
        return self.tname == "F"

    def is_long(self) -> bool:
        return self.tname == "J"

    def is_short(self) -> bool:
        return self.tname == "I"

    def is_double_or_long(self) -> bool:
        return self.tname in {"D", "J"}

    def is_reference(self) -> bool:
        return (
            self.tname == "R"
            or self.tname.startswith("L")
            or self.tname.startswith("[")
        )

    def is_array(self) -> bool:
        return self.tname == "R" or self.tname.startswith("[")

    def can_assign_to(self, other: "BasicValue") -> bool:
        if self.is_reference() and other.is_reference():
            # don't check for arrays and exact types of references
            return True
        if self.is_int() and other.is_int():
            # don't check exact int/short/boolean/byte/char types
            return True
        if self.tname == other.tname:
            # primitive types
            return True
        return False

    def get_name(self) -> str:
        if self.is_reference():
            return Insn.class_i2qname(self.tname[1:-1])
            # return re_fqname.sub(
            #     r"\g<canonical>", re_cdesc.sub(r"L\g<canonical>;", self.tname)
            # )
        else:
            return self.tname


@dataclasses.dataclass
class InsnConstraint:
    values: List[BasicValue] = dataclasses.field(default_factory=list)
    locals: List[BasicValue] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        s = f"InsnConstraint(values={self.values}, locals={{"
        for i, v in enumerate(self.locals):
            if not v.is_uninitialized():
                s += f"{i}: {v}, "
        s += "})"
        return s


re_any_int = re.compile(r"-?[0-9]+")
re_any_float = re.compile(r"(-?[0-9]*\.?[0-9]+(E-?[0-9]+)?)|NaN|-?Infinity")


@dataclasses.dataclass
class TokConstraint:
    allowed: Set[str] = None
    allowed_trie: Trie = None
    any_ident: bool = False
    any_spident: bool = False
    any_int: bool = False
    any_float: bool = False
    soft_any_ident: bool = False
    soft_any_spident: bool = False

    def accept(self, tok: str) -> bool:
        if self.any_ident:
            # no need to check exact content for now because they're constrained from previous steps
            return True
        if self.any_int:
            return re_any_int.fullmatch(tok)
        if self.any_float:
            return re_any_float.fullmatch(tok)

        if self.any_spident:
            return tok in {"<init>", "<clinit>"}

        if self.allowed is not None:
            return tok in self.allowed
        if self.allowed_trie is not None:
            return self.allowed_trie.has_key(tok)


class Operand:
    def __init__(self):
        self.toks: List[str] = []

    def fork(self) -> "Operand":
        # child class should override if they introduce new stateful fields
        ret = copy.copy(self)
        ret.toks = list(self.toks)
        return ret

    @abc.abstractmethod
    def accept_toks(self, toks: List[str]) -> bool:
        raise NotImplementedError()

    def is_completed(self) -> bool:
        return self.accept_toks(self.toks)

    def accept_tok(self, tok: str) -> bool:
        return self.accept_toks(self.toks + [tok])

    def submit_tok(self, tok: str) -> None:
        self.toks.append(tok)

    @abc.abstractmethod
    def get_next_constraint(self) -> TokConstraint:
        raise NotImplementedError()

    def get_next_constraint_after_tok(self, tok: str) -> TokConstraint:
        self.toks.append(tok)
        ret = self.get_next_constraint()
        del self.toks[-1]
        return ret

    def __repr__(self) -> str:
        return f"{type(self).__name__}(toks={self.toks})"


class IntOperand(Operand):
    def __init__(self, allowed: Set[int] = None, any_int: bool = False):
        super().__init__()
        self.allowed = [str(x) for x in allowed] if allowed is not None else None
        self.any_int = any_int

    def accept_toks(self, toks: List[str]) -> bool:
        return len(toks) == 1

    def get_next_constraint(self) -> TokConstraint:
        return TokConstraint(allowed=self.allowed, any_int=self.any_int)


class NameOperand(Operand):
    def __init__(self, allow_special: bool = False):
        super().__init__()
        self.allow_special = allow_special

    def accept_toks(self, toks: List[str]) -> bool:
        return len(toks) == 1

    def get_next_constraint(self) -> TokConstraint:
        return TokConstraint(any_ident=True, any_spident=self.allow_special)


class ClassDescOperand(Operand):
    def __init__(self):
        super().__init__()

    def accept_toks(self, toks: List[str]) -> bool:
        i = 0
        while i < len(toks) and toks[i] == "[":
            i += 1

        if i == len(toks):
            return False
        if toks[i] == "L":
            return len(toks) == i + 3 and toks[i + 2] == ";"
        else:
            return toks[i] in {"B", "C", "D", "F", "I", "J", "S", "Z"}

    def get_next_constraint(self) -> TokConstraint:
        if len(self.toks) == 0 or self.toks[-1] == "[":
            return TokConstraint(
                allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
            )
        elif self.toks[-1] == "L":
            return TokConstraint(any_ident=True)
        else:
            return TokConstraint(allowed={";"})


class MethodDescOperand(Operand):
    def __init__(self, is_static: bool, values: List[BasicValue]):
        super().__init__()
        self.is_static = is_static
        self.values = values

    def accept_toks(self, toks: List[str]) -> bool:
        if len(toks) == 0 or toks[0] != "(":
            return False

        i = 1
        while i < len(toks) and toks[i] != ")":
            if toks[i] == "L":
                if len(toks) < i + 3 or toks[i + 2] != ";" or toks[i + 1] == ")":
                    return False
                i += 3
            else:
                if toks[i] not in {"[", "B", "C", "D", "F", "I", "J", "S", "Z"}:
                    return False
                i += 1

        if i >= len(toks) or toks[i] != ")":
            return False
        i += 1

        while i < len(toks) and toks[i] == "[":
            i += 1

        if i == len(toks):
            return False
        if toks[i] == "L":
            return len(toks) == i + 3 and toks[i + 2] == ";"
        else:
            return toks[i] in {"B", "C", "D", "F", "I", "J", "S", "Z", "V"}

    def get_next_constraint(self) -> TokConstraint:
        if len(self.toks) == 0:
            return TokConstraint(allowed="(")

        if ")" not in self.toks:
            # parameters' types
            if self.toks[-1] == "[":
                return TokConstraint(
                    allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
                )
            elif self.toks[-1] == "L":
                return TokConstraint(any_ident=True)
            elif len(self.toks) > 2 and self.toks[-2] == "L":
                return TokConstraint(allowed={";"})
            else:
                # starting a new parameter: check against the values types
                return TokConstraint(
                    allowed=self.get_continuation(
                        self.get_consumed_types(self.toks[1:])
                    )
                )
        else:
            # return value's type
            if self.toks[-1] == ")":
                return TokConstraint(
                    allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z", "V"}
                )
            elif self.toks[-1] == "[":
                return TokConstraint(
                    allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
                )
            elif self.toks[-1] == "L":
                return TokConstraint(any_ident=True)
            else:
                return TokConstraint(allowed={";"})

    def get_consumed_types(self, toks: List[str]) -> List[BasicValue]:
        consumed = []
        if not self.is_static:
            consumed.append(BasicValue("R"))
        for tname in postprocess_recover_break_down_desc(toks):
            consumed.append(BasicValue(tname))
        return consumed

    def get_continuation(self, consumed: List[BasicValue]) -> List[str]:
        allowed = set()

        for i in range(len(self.values) - len(consumed) + 1):
            compatible = True
            for j in range(len(consumed)):
                if not self.values[i + j].can_assign_to(consumed[j]):
                    compatible = False
                    break

            if compatible:
                if i + len(consumed) < len(self.values):
                    v = self.values[i + len(consumed)]
                    if v.is_reference():
                        allowed |= {"[", "L"}
                    elif v.is_int():
                        allowed |= {"B", "C", "I", "S", "Z"}
                    else:
                        allowed.add(v.tname)
                else:
                    allowed.add(")")

        return allowed


class LdcOperand(Operand):
    def __init__(self, cnames: Optional[Trie] = None):
        super().__init__()
        self.cnames = cnames

    def accept_toks(self, toks: List[str]) -> bool:
        # TODO: assuming string literals are masked
        if len(toks) < 3 or toks[-1] != "LDC_END":
            return False

        if toks[0] == "Type":
            i = 1
            while i < len(toks) and toks[i] == "[":
                i += 1
            if i == len(toks):
                return False
            if toks[i] == "L":
                return len(toks) == i + 4 and toks[i + 2] == ";"
            else:
                return toks[i] in {"B", "C", "D", "F", "I", "J", "S", "Z"}
        else:
            return toks[0] in {"String", "Integer", "Float", "Long", "Double"}

    def get_next_constraint(self) -> TokConstraint:
        if len(self.toks) == 0:
            return TokConstraint(
                allowed={"Type", "String", "Integer", "Float", "Long", "Double"}
            )
        elif len(self.toks) == 1:
            if self.toks[-1] == "String":
                return TokConstraint(allowed={"STR"})
            elif self.toks[-1] in {"Integer", "Long"}:
                return TokConstraint(any_int=True)
            elif self.toks[-1] in {"Float", "Double"}:
                return TokConstraint(any_float=True)
            elif self.toks[-1] == "Type":
                return TokConstraint(allowed={"[", "L"})
            else:
                raise RuntimeError("should not happen")
        elif self.toks[-1] == "[":
            return TokConstraint(
                allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
            )
        elif self.toks[-1] == "L":
            if self.cnames is None:
                return TokConstraint(any_ident=True)
            else:
                return TokConstraint(allowed_trie=self.cnames)
        elif len(self.toks) > 2 and self.toks[-2] == "L":
            return TokConstraint(allowed={";"})
        else:
            return TokConstraint(allowed={"LDC_END"})


class TypeOperand(Operand):
    def __init__(self, allow_array: bool = True, cnames: Optional[Trie] = None):
        super().__init__()
        self.allow_array = allow_array
        self.cnames = cnames

    def accept_toks(self, toks: List[str]) -> bool:
        if self.allow_array:
            i = 0
            while i < len(toks) and toks[i] == "[":
                i += 1

            if i == len(toks):
                return False
            elif len(toks) == 1:
                return True
            elif toks[i] == "L":
                return len(toks) == i + 3 and toks[i + 2] == ";"
            else:
                return toks[i] in {"B", "C", "D", "F", "I", "J", "S", "Z"}
        else:
            return len(toks) == 1

    def get_next_constraint(self) -> TokConstraint:
        if self.allow_array:
            if len(self.toks) == 0:
                return TokConstraint(allowed={"["}, any_ident=True)
            elif self.toks[-1] == "[":
                return TokConstraint(
                    allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
                )
            elif self.toks[-1] == "L":
                if self.cnames is None:
                    return TokConstraint(any_ident=True)
                else:
                    return TokConstraint(allowed_trie=self.cnames, soft_any_ident=True)
            elif len(self.toks) > 2 and self.toks[-2] == "L":
                return TokConstraint(allowed={";"})
            else:
                raise RuntimeError("should not happen")
        else:
            return TokConstraint(any_ident=True)


class FieldConstrainedRefOperand(Operand):
    def __init__(
        self,
        cnames: Trie,
        cname2fnames: Dict[str, Trie],
        cf2fs: Dict[Tuple[str, str], List[FieldStructure]],
    ):
        super().__init__()
        self.cnames = cnames
        self.cname2fnames = cname2fnames
        self.cf2fs = cf2fs

    def accept_toks(self, toks: List[str]) -> bool:
        # first 2 tokens: class_name field_name
        if len(toks) <= 2:
            return False

        i = 2
        while i < len(toks) and toks[i] == "[":
            i += 1

        if i == len(toks):
            return False
        if toks[i] == "L":
            return len(toks) == i + 3 and toks[i + 2] == ";"
        else:
            return toks[i] in {"B", "C", "D", "F", "I", "J", "S", "Z"}

    @functools.lru_cache()
    def get_possible_desc_tokens(self, cname: str, fname: str) -> Set[Tuple[str]]:
        ret = set()
        for fs in self.cf2fs.get((cname, fname), set()):
            desc = Insn.class_name2desc(fs.type.split(".")[-1])
            ret.add(tuple(preprocess_break_down_desc([desc])))
        return ret

    def get_next_constraint(self) -> TokConstraint:
        if len(self.toks) == 0:
            return TokConstraint(allowed_trie=self.cnames, soft_any_ident=True)
        elif len(self.toks) == 1:
            if self.toks[0] in self.cname2fnames:
                return TokConstraint(
                    allowed_trie=self.cname2fnames[self.toks[0]], soft_any_ident=True
                )
            else:
                return TokConstraint(soft_any_ident=True)
        elif (self.toks[0], self.toks[1]) in self.cf2fs:
            len_cur_desc = len(self.toks) - 2
            allowed = set()
            for desc in self.get_possible_desc_tokens(self.toks[0], self.toks[1]):
                if len(desc) > len_cur_desc and desc[:len_cur_desc] == tuple(
                    self.toks[2:]
                ):
                    allowed.add(desc[len_cur_desc])
            return TokConstraint(allowed=allowed)
        else:
            # not found in cf2fs, back off to match any ClassDesc
            if len(self.toks) == 2 or self.toks[-1] == "[":
                return TokConstraint(
                    allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
                )
            elif self.toks[-1] == "L":
                return TokConstraint(any_ident=True)
            else:
                return TokConstraint(allowed={";"})


class MethodConstrainedRefOperand(Operand):
    def __init__(
        self,
        cnames: Trie,
        cname2mnames: Dict[str, Trie],
        cm2ms: Dict[Tuple[str, str], List[MethodStructure]],
        is_static: bool,
        values: List[BasicValue],
    ):
        super().__init__()
        self.cnames = cnames
        self.cname2mnames = cname2mnames
        self.cm2ms = cm2ms
        self.is_static = is_static
        self.values = values

    def accept_toks(self, toks: List[str]) -> bool:
        # first 2 tokens: class_name method_name
        if len(toks) <= 2:
            return False

        i = 2
        if toks[i] != "(":
            return False

        i += 1
        while i < len(toks) and toks[i] != ")":
            if toks[i] == "L":
                if len(toks) < i + 3 or toks[i + 2] != ";" or toks[i + 1] == ")":
                    return False
                i += 3
            else:
                if toks[i] not in {"[", "B", "C", "D", "F", "I", "J", "S", "Z"}:
                    return False
                i += 1

        if i >= len(toks) or toks[i] != ")":
            return False
        i += 1

        while i < len(toks) and toks[i] == "[":
            i += 1

        if i == len(toks):
            return False
        if toks[i] == "L":
            return len(toks) == i + 3 and toks[i + 2] == ";"
        else:
            return toks[i] in {"B", "C", "D", "F", "I", "J", "S", "Z", "V"}

    @functools.lru_cache()
    def get_possible_desc_tokens(self, cname: str, fname: str) -> Set[Tuple[str]]:
        ret = set()
        for ms in self.cm2ms.get((cname, fname), set()):
            # check if this method can pass type checking
            if len(ms.ptypes) + (0 if self.is_static else 1) > len(self.values):
                continue

            pdescs = [Insn.class_name2desc(ptype.split(".")[-1]) for ptype in ms.ptypes]

            pvalues = []
            if not self.is_static:
                pvalues.append(BasicValue("R"))
            for pdesc in pdescs:
                pvalues.append(BasicValue(pdesc))

            compatible = True
            for pvalue, value in zip(pvalues, self.values[-len(pvalues) :]):
                if not value.can_assign_to(pvalue):
                    compatible = False
                    break
            if not compatible:
                continue

            # assemble the tokens of this method desc
            rdesc = Insn.class_name2desc(ms.rtype.split(".")[-1])
            ret.add(tuple(preprocess_break_down_desc(["(", *pdescs, ")", rdesc])))
        return ret

    def get_next_constraint(self) -> TokConstraint:
        if len(self.toks) == 0:
            return TokConstraint(allowed_trie=self.cnames, soft_any_ident=True)
        elif len(self.toks) == 1:
            if self.toks[0] in self.cname2mnames:
                return TokConstraint(
                    allowed_trie=self.cname2mnames[self.toks[0]],
                    soft_any_ident=True,
                    soft_any_spident=True,
                )
            else:
                return TokConstraint(soft_any_ident=True, soft_any_spident=True)
        elif (self.toks[0], self.toks[1]) in self.cm2ms:
            len_cur_desc = len(self.toks) - 2
            allowed = set()
            for desc in self.get_possible_desc_tokens(self.toks[0], self.toks[1]):
                if len(desc) > len_cur_desc and desc[:len_cur_desc] == tuple(
                    self.toks[2:]
                ):
                    allowed.add(desc[len_cur_desc])
            return TokConstraint(allowed=allowed)
        else:
            # not found in cm2ms, back off to match any MethodDesc
            if len(self.toks) == 2:
                return TokConstraint(allowed="(")

            if ")" not in self.toks:
                # parameters' types
                if self.toks[-1] == "[":
                    return TokConstraint(
                        allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
                    )
                elif self.toks[-1] == "L":
                    return TokConstraint(any_ident=True)
                elif len(self.toks) > 4 and self.toks[-2] == "L":
                    return TokConstraint(allowed={";"})
                else:
                    # starting a new parameter: check against the values types
                    return TokConstraint(
                        allowed=self.get_continuation(
                            self.get_consumed_types(self.toks[3:])
                        )
                    )
            else:
                # return value's type
                if self.toks[-1] == ")":
                    return TokConstraint(
                        allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z", "V"}
                    )
                elif self.toks[-1] == "[":
                    return TokConstraint(
                        allowed={"[", "L", "B", "C", "D", "F", "I", "J", "S", "Z"}
                    )
                elif self.toks[-1] == "L":
                    return TokConstraint(any_ident=True)
                else:
                    return TokConstraint(allowed={";"})

    def get_consumed_types(self, toks: List[str]) -> List[BasicValue]:
        consumed = []
        if not self.is_static:
            consumed.append(BasicValue("R"))
        for tname in postprocess_recover_break_down_desc(toks):
            consumed.append(BasicValue(tname))
        return consumed

    def get_continuation(self, consumed: List[BasicValue]) -> List[str]:
        allowed = set()

        for i in range(len(self.values) - len(consumed) + 1):
            compatible = True
            for j in range(len(consumed)):
                if not self.values[i + j].can_assign_to(consumed[j]):
                    compatible = False
                    break

            if compatible:
                if i + len(consumed) < len(self.values):
                    v = self.values[i + len(consumed)]
                    if v.is_reference():
                        allowed |= {"[", "L"}
                    elif v.is_int():
                        allowed |= {"B", "C", "I", "S", "Z"}
                    else:
                        allowed.add(v.tname)
                else:
                    allowed.add(")")

        return allowed
