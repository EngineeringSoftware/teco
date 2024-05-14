import inspect
from typing import List

from teco.data.structures import (
    ClassStructure,
    Consts,
    FieldStructure,
    Insn,
    MethodStructure,
    Scope,
)


def method_insn(op: str, cs: ClassStructure, ms: MethodStructure) -> Insn:
    return Insn(op=op, operands=[cs.iname, ms.name, ms.desc])


def field_insn(op: str, cs: ClassStructure, fs: FieldStructure) -> Insn:
    return Insn(op=op, operands=[cs.iname, fs.name])


class PesudoStructures:

    reserved_cid_begin = 10_000_000
    reserved_mid_begin = 10_000_000
    reserved_fid_begin = 10_000_000
    cid_offset = 0
    mid_offset = 0
    fid_offset = 0

    @classmethod
    def get_classes(cls) -> List[ClassStructure]:
        classes = []
        for name, attr in inspect.getmembers(cls, lambda a: not (inspect.isroutine(a))):
            if name.startswith("c_"):
                classes.append(attr)
        classes.sort(key=lambda x: x.id)
        return classes

    @classmethod
    def get_methods(cls) -> List[MethodStructure]:
        methods = []
        for name, attr in inspect.getmembers(cls, lambda a: not (inspect.isroutine(a))):
            if name.startswith("m_"):
                methods.append(attr)
        methods.sort(key=lambda x: x.id)
        return methods

    @classmethod
    def get_fields(cls) -> List[FieldStructure]:
        fields = []
        for name, attr in inspect.getmembers(cls, lambda a: not (inspect.isroutine(a))):
            if name.startswith("f_"):
                fields.append(attr)
        fields.sort(key=lambda x: x.id)
        return fields

    c_array: ClassStructure
    f_array_length: FieldStructure
    get_array_length: Insn
    m_array_new: MethodStructure
    inv_array_new: Insn
    m_array_store: MethodStructure
    inv_array_store: Insn
    m_array_load: MethodStructure
    inv_array_load: Insn

    c_arithmetic: ClassStructure
    m_arithmetic_add: MethodStructure
    inv_arithmetic_add: Insn
    m_arithmetic_sub: MethodStructure
    inv_arithmetic_sub: Insn
    m_arithmetic_mul: MethodStructure
    inv_arithmetic_mul: Insn
    m_arithmetic_div: MethodStructure
    inv_arithmetic_div: Insn
    m_arithmetic_rem: MethodStructure
    inv_arithmetic_rem: Insn
    m_arithmetic_neg: MethodStructure
    inv_arithmetic_neg: Insn
    m_arithmetic_and: MethodStructure
    inv_arithmetic_and: Insn
    m_arithmetic_or: MethodStructure
    inv_arithmetic_or: Insn
    m_arithmetic_xor: MethodStructure
    inv_arithmetic_xor: Insn
    m_arithmetic_shl: MethodStructure
    inv_arithmetic_shl: Insn
    m_arithmetic_shr: MethodStructure
    inv_arithmetic_shr: Insn
    m_arithmetic_ushr: MethodStructure
    inv_arithmetic_ushr: Insn
    m_arithmetic_inc: MethodStructure
    inv_arithmetic_inc: Insn
    m_arithmetic_dec: MethodStructure
    inv_arithmetic_dec: Insn
    m_arithmetic_incx: MethodStructure
    inv_arithmetic_incx: Insn
    m_arithmetic_cmp: MethodStructure
    inv_arithmetic_cmp: Insn

    c_typing: ClassStructure
    m_typing_instanceof: MethodStructure
    inv_typing_instanceof: Insn

    c_cond: ClassStructure
    m_cond_ne_b: MethodStructure
    inv_cond_ne_b: Insn
    m_cond_eq_b: MethodStructure
    inv_cond_eq_b: Insn
    m_cond_lt_b: MethodStructure
    inv_cond_lt_b: Insn
    m_cond_le_b: MethodStructure
    inv_cond_le_b: Insn
    m_cond_gt_b: MethodStructure
    inv_cond_gt_b: Insn
    m_cond_ge_b: MethodStructure
    inv_cond_ge_b: Insn
    m_cond_ne0_b: MethodStructure
    inv_cond_ne0_b: Insn
    m_cond_eq0_b: MethodStructure
    inv_cond_eq0_b: Insn
    m_cond_lt0_b: MethodStructure
    inv_cond_lt0_b: Insn
    m_cond_le0_b: MethodStructure
    inv_cond_le0_b: Insn
    m_cond_gt0_b: MethodStructure
    inv_cond_gt0_b: Insn
    m_cond_ge0_b: MethodStructure
    inv_cond_ge0_b: Insn
    m_cond_eqnull_b: MethodStructure
    inv_cond_eqnull_b: Insn
    m_cond_nenull_b: MethodStructure
    inv_cond_nenull_b: Insn

    @classmethod
    def initialize(cls):
        cls.cid_offset = 0
        cls.mid_offset = 0
        cls.fid_offset = 0

        # --------------------
        # arrays
        # --------------------

        cls.c_array = ClassStructure(
            id=cls.reserved_cid_begin + cls.cid_offset,
            access=Consts.acc_public,
            scope=Scope.JRE,
            name="pesudo.Array",
        )
        cls.cid_offset += 1

        cls.f_array_length = FieldStructure(
            id=cls.reserved_fid_begin + cls.fid_offset,
            access=Consts.acc_public,
            clz=cls.c_array.id,
            name="length",
            type="int",
        )
        cls.fid_offset += 1
        cls.c_array.fields.append(cls.f_array_length.id)
        cls.get_array_length = field_insn("GETFIELD", cls.c_array, cls.f_array_length)

        cls.m_array_new = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_array.id,
            name="new",
            ptypes=["int"],
            rtype="pesudo.Array",
        )
        cls.mid_offset += 1
        cls.c_array.methods.append(cls.m_array_new.id)
        cls.inv_array_new = method_insn("INVOKESTATIC", cls.c_array, cls.m_array_new)

        cls.m_array_store = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public,
            clz=cls.c_array.id,
            name="store",
            ptypes=["int", "any"],
            rtype="pesudo.Array",
        )
        cls.mid_offset += 1
        cls.c_array.methods.append(cls.m_array_store.id)
        cls.inv_array_store = method_insn(
            "INVOKEVIRTUAL", cls.c_array, cls.m_array_store
        )

        cls.m_array_load = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public,
            clz=cls.c_array.id,
            name="load",
            ptypes=["int"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_array.methods.append(cls.m_array_load.id)
        cls.inv_array_load = method_insn("INVOKEVIRTUAL", cls.c_array, cls.m_array_load)

        # --------------------
        # arithmetic
        # --------------------

        cls.c_arithmetic = ClassStructure(
            id=cls.reserved_cid_begin + cls.cid_offset,
            access=Consts.acc_public,
            scope=Scope.JRE,
            name="pesudo.Arithmetic",
        )
        cls.cid_offset += 1

        cls.m_arithmetic_add = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="+",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_add.id)
        cls.inv_arithmetic_add = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_add
        )

        cls.m_arithmetic_sub = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="-",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_sub.id)
        cls.inv_arithmetic_sub = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_sub
        )

        cls.m_arithmetic_mul = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="*",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_mul.id)
        cls.inv_arithmetic_mul = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_mul
        )

        cls.m_arithmetic_div = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="/",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_div.id)
        cls.inv_arithmetic_div = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_div
        )

        cls.m_arithmetic_rem = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="%",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_rem.id)
        cls.inv_arithmetic_rem = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_rem
        )

        cls.m_arithmetic_neg = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="-",
            ptypes=["number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_neg.id)
        cls.inv_arithmetic_neg = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_neg
        )

        cls.m_arithmetic_and = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="&",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_and.id)
        cls.inv_arithmetic_and = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_and
        )

        cls.m_arithmetic_or = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="|",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_or.id)
        cls.inv_arithmetic_or = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_or
        )

        cls.m_arithmetic_xor = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="^",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_xor.id)
        cls.inv_arithmetic_xor = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_xor
        )

        cls.m_arithmetic_shl = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="<<",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_shl.id)
        cls.inv_arithmetic_shl = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_shl
        )

        cls.m_arithmetic_shr = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name=">>",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_shr.id)
        cls.inv_arithmetic_shr = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_shr
        )

        cls.m_arithmetic_ushr = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name=">>>",
            ptypes=["number", "number"],
            rtype="number",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_ushr.id)
        cls.inv_arithmetic_ushr = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_ushr
        )

        cls.m_arithmetic_inc = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="++",
            ptypes=["int"],
            rtype="void",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_inc.id)
        cls.inv_arithmetic_inc = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_inc
        )

        cls.m_arithmetic_dec = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="--",
            ptypes=["int"],
            rtype="void",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_dec.id)
        cls.inv_arithmetic_dec = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_dec
        )

        cls.m_arithmetic_incx = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="+=",
            ptypes=["int", "int"],
            rtype="void",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_incx.id)
        cls.inv_arithmetic_incx = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_incx
        )

        cls.m_arithmetic_cmp = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_arithmetic.id,
            name="compare",
            ptypes=["number", "number"],
            rtype="int",
        )
        cls.mid_offset += 1
        cls.c_arithmetic.methods.append(cls.m_arithmetic_cmp.id)
        cls.inv_arithmetic_cmp = method_insn(
            "INVOKESTATIC", cls.c_arithmetic, cls.m_arithmetic_cmp
        )

        # --------------------
        # Typing
        # --------------------

        cls.c_typing = ClassStructure(
            id=cls.reserved_cid_begin + cls.cid_offset,
            access=Consts.acc_public,
            scope=Scope.JRE,
            name="pesudo.Typing",
        )
        cls.cid_offset += 1

        cls.m_typing_instanceof = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_typing.id,
            name="instanceof",
            ptypes=["java.lang.Object", "type"],
            rtype="boolean",
        )
        cls.mid_offset += 1
        cls.c_typing.methods.append(cls.m_typing_instanceof.id)
        cls.inv_typing_instanceof = method_insn(
            "INVOKESTATIC", cls.c_typing, cls.m_typing_instanceof
        )

        # --------------------
        # Conditional expression
        # --------------------

        cls.c_cond = ClassStructure(
            id=cls.reserved_cid_begin + cls.cid_offset,
            access=Consts.acc_public,
            scope=Scope.JRE,
            name="pesudo.Conditional",
        )
        cls.cid_offset += 1

        cls.m_cond_ne_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="!=",
            ptypes=["any", "any", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_ne_b.id)
        cls.inv_cond_ne_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_ne_b)

        cls.m_cond_eq_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="==",
            ptypes=["any", "any", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_eq_b.id)
        cls.inv_cond_eq_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_eq_b)

        cls.m_cond_lt_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="<",
            ptypes=["any", "any", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_lt_b.id)
        cls.inv_cond_lt_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_lt_b)

        cls.m_cond_le_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="<=",
            ptypes=["any", "any", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_le_b.id)
        cls.inv_cond_le_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_le_b)

        cls.m_cond_gt_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name=">",
            ptypes=["any", "any", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_gt_b.id)
        cls.inv_cond_gt_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_gt_b)

        cls.m_cond_ge_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name=">=",
            ptypes=["any", "any", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_ge_b.id)
        cls.inv_cond_ge_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_ge_b)

        cls.m_cond_ne0_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="!=0",
            ptypes=["int", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_ne0_b.id)
        cls.inv_cond_ne0_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_ne0_b)

        cls.m_cond_eq0_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="==0",
            ptypes=["int", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_eq0_b.id)
        cls.inv_cond_eq0_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_eq0_b)

        cls.m_cond_lt0_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="<0",
            ptypes=["int", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_lt0_b.id)
        cls.inv_cond_lt0_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_lt0_b)

        cls.m_cond_le0_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="<=0",
            ptypes=["int", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_le0_b.id)
        cls.inv_cond_le0_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_le0_b)

        cls.m_cond_gt0_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name=">0",
            ptypes=["int", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_gt0_b.id)
        cls.inv_cond_gt0_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_gt0_b)

        cls.m_cond_ge0_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name=">=0",
            ptypes=["int", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_ge0_b.id)
        cls.inv_cond_ge0_b = method_insn("INVOKESTATIC", cls.c_cond, cls.m_cond_ge0_b)

        cls.m_cond_eqnull_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="==null",
            ptypes=["java.lang.Object", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_eqnull_b.id)
        cls.inv_cond_eqnull_b = method_insn(
            "INVOKESTATIC", cls.c_cond, cls.m_cond_eqnull_b
        )

        cls.m_cond_nenull_b = MethodStructure(
            id=cls.reserved_mid_begin + cls.mid_offset,
            access=Consts.acc_public | Consts.acc_static,
            clz=cls.c_cond.id,
            name="!=null",
            ptypes=["java.lang.Object", "any", "any"],
            rtype="any",
        )
        cls.mid_offset += 1
        cls.c_cond.methods.append(cls.m_cond_nenull_b.id)
        cls.inv_cond_nenull_b = method_insn(
            "INVOKESTATIC", cls.c_cond, cls.m_cond_nenull_b
        )


PesudoStructures.initialize()


if __name__ == "__main__":
    classes = PesudoStructures.get_classes()
    methods = PesudoStructures.get_methods()
    fields = PesudoStructures.get_fields()
    print(
        f"Defined {len(classes)} classes, {len(methods)} methods, {len(fields)} fields"
    )
    for cs in classes:
        print(cs)
    for ms in methods:
        print(ms)
    for fs in fields:
        print(fs)
