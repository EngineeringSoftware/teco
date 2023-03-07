import abc
import dataclasses
from typing import Dict, List, Optional, Tuple, Union


@dataclasses.dataclass
class Subtokenization:
    # encoded subtokens
    esubtoks: List[str] = dataclasses.field(default_factory=list)
    # mapping from token indice to subtoken beg/end indices
    t2st: Dict[int, Tuple[int, int]] = dataclasses.field(default_factory=dict)
    # mapping from subtoken indice to token indice (could be None)
    st2t: Dict[int, Optional[int]] = dataclasses.field(default_factory=dict)

    @classmethod
    def combine(cls, subtokenizations: List["Subtokenization"]) -> "Subtokenization":
        """Combine multiple subtokenizations into one."""
        ret = cls(subtoks=[], t2st={}, st2t={})

        t_offset = 0
        st_offset = 0
        for s in subtokenizations:
            ret.esubtoks += s.esubtoks
            for t, (st_beg, st_end) in s.t2st.items():
                ret.t2st[t + t_offset] = (st_beg + st_offset, st_end + st_offset)
            for st, t in s.st2t.items():
                ret.st2t[st + st_offset] = t + t_offset if t is not None else None

            t_offset += len(s.t2st)
            st_offset += len(s.st2t)

        return ret


class Subtokenizer:

    SIMP_INSN_TOKENS = [
        " <M>",
        " <C>",
        " <pL>",
        " <gL>",
        " <pF>",
        " <gF>",
        " <decl>",
        " <ret>",
        " <throw>",
        " <multi>",
    ]

    INSN_TOKENS = [
        " AALOAD",
        " AASTORE",
        " ACONST_NULL",
        " ALOAD",
        " ALOAD_0",
        " ALOAD_1",
        " ALOAD_2",
        " ALOAD_3",
        " ANEWARRAY",
        " ARETURN",
        " ARRAYLENGTH",
        " ASTORE",
        " ASTORE_0",
        " ASTORE_1",
        " ASTORE_2",
        " ASTORE_3",
        " ATHROW",
        " BALOAD",
        " BASTORE",
        " BIPUSH",
        " CALOAD",
        " CASTORE",
        " CHECKCAST",
        " D2F",
        " D2I",
        " D2L",
        " DADD",
        " DALOAD",
        " DASTORE",
        " DCMPG",
        " DCMPL",
        " DCONST_0",
        " DCONST_1",
        " DDIV",
        " DLOAD",
        " DLOAD_0",
        " DLOAD_1",
        " DLOAD_2",
        " DLOAD_3",
        " DMUL",
        " DNEG",
        " DREM",
        " DRETURN",
        " DSTORE",
        " DSTORE_0",
        " DSTORE_1",
        " DSTORE_2",
        " DSTORE_3",
        " DSUB",
        " DUP",
        " DUP_X1",
        " DUP_X2",
        " DUP2",
        " DUP2_X1",
        " DUP2_X2",
        " F2D",
        " F2I",
        " F2L",
        " FADD",
        " FALOAD",
        " FASTORE",
        " FCMPG",
        " FCMPL",
        " FCONST_0",
        " FCONST_1",
        " FCONST_2",
        " FDIV",
        " FLOAD",
        " FLOAD_0",
        " FLOAD_1",
        " FLOAD_2",
        " FLOAD_3",
        " FMUL",
        " FNEG",
        " FREM",
        " FRETURN",
        " FSTORE",
        " FSTORE_0",
        " FSTORE_1",
        " FSTORE_2",
        " FSTORE_3",
        " FSUB",
        " GETFIELD",
        " GETSTATIC",
        " GOTO",
        " GOTO_W",
        " I2B",
        " I2C",
        " I2D",
        " I2F",
        " I2L",
        " I2S",
        " IADD",
        " IALOAD",
        " IAND",
        " IASTORE",
        " ICONST_M1",
        " ICONST_0",
        " ICONST_1",
        " ICONST_2",
        " ICONST_3",
        " ICONST_4",
        " ICONST_5",
        " IDIV",
        " IF_ACMPEQ",
        " IF_ACMPNE",
        " IF_ICMPEQ",
        " IF_ICMPGE",
        " IF_ICMPGT",
        " IF_ICMPLE",
        " IF_ICMPLT",
        " IF_ICMPNE",
        " IFEQ",
        " IFGE",
        " IFGT",
        " IFLE",
        " IFLT",
        " IFNE",
        " IFNONNULL",
        " IFNULL",
        " IINC",
        " ILOAD",
        " ILOAD_0",
        " ILOAD_1",
        " ILOAD_2",
        " ILOAD_3",
        " IMUL",
        " INEG",
        " INSTANCEOF",
        " INVOKEDYNAMIC",
        " INVOKEINTERFACE",
        " INVOKESPECIAL",
        " INVOKESTATIC",
        " INVOKEVIRTUAL",
        " IOR",
        " IREM",
        " IRETURN",
        " ISHL",
        " ISHR",
        " ISTORE",
        " ISTORE_0",
        " ISTORE_1",
        " ISTORE_2",
        " ISTORE_3",
        " ISUB",
        " IUSHR",
        " IXOR",
        " JSR",
        " JSR_W",
        " L2D",
        " L2F",
        " L2I",
        " LADD",
        " LALOAD",
        " LAND",
        " LASTORE",
        " LCMP",
        " LCONST_0",
        " LCONST_1",
        " LDC",
        " LDC_W",
        " LDC2_W",
        " LDIV",
        " LLOAD",
        " LLOAD_0",
        " LLOAD_1",
        " LLOAD_2",
        " LLOAD_3",
        " LMUL",
        " LNEG",
        " LOOKUPSWITCH",
        " LOR",
        " LREM",
        " LRETURN",
        " LSHL",
        " LSHR",
        " LSTORE",
        " LSTORE_0",
        " LSTORE_1",
        " LSTORE_2",
        " LSTORE_3",
        " LSUB",
        " LUSHR",
        " LXOR",
        " MONITORENTER",
        " MONITOREXIT",
        " MULTIANEWARRAY",
        " NEW",
        " NEWARRAY",
        " NOP",
        " POP",
        " POP2",
        " PUTFIELD",
        " PUTSTATIC",
        " RET",
        " RETURN",
        " SALOAD",
        " SASTORE",
        " SIPUSH",
        " SWAP",
        " TABLESWITCH",
        " WIDE",
        " LDC_END",
    ]

    def __init__(self):
        self.bos_token: str = None
        self.eos_token: str = None
        self.unk_token: str = None
        self.pad_token: str = None
        self.cls_token: str = None
        self.sep_token: str = None
        self.mask_token: str = None
        self.bos_token_id: int = None
        self.eos_token_id: int = None
        self.unk_token_id: int = None
        self.pad_token_id: int = None
        self.cls_token_id: int = None
        self.sep_token_id: int = None
        self.mask_token_id: int = None

    @abc.abstractmethod
    def toks2subtokenization(self, tokens: List[str]) -> Subtokenization:
        raise NotImplementedError()

    def toks2esubtoks(self, tokens: List[str]) -> List[str]:
        return self.toks2subtokenization(tokens).esubtoks

    @abc.abstractmethod
    def str2esubtoks(self, s: str) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def esubtok2id(self, subtoks: Union[List[str], str]) -> Union[List[int], int]:
        raise NotImplementedError()

    def toks2stids(self, tokens: List[str]) -> List[int]:
        return self.esubtok2id(self.toks2esubtoks(tokens))

    @abc.abstractmethod
    def id2subtok(self, ids: Union[List[int], int]) -> Union[List[str], str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def id2esubtok(self, ids: Union[List[int], int]) -> Union[List[str], str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_vocab(exclude_special_tokens: bool = True) -> Dict[str, int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def is_special_token(self, token: str) -> bool:
        return token in {
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.pad_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
        }

    def is_special_token_id(self, token_id: int) -> bool:
        return token_id in {
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        }
