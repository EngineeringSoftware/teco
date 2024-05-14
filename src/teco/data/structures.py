import copy
import dataclasses
import enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import seutil as su

logger = su.log.get_logger(__name__)


class Scope(enum.Enum):
    APP = "APP"
    TEST = "TEST"
    LIB = "LIB"
    JRE = "JRE"


class Consts:
    # ===== source code =====

    # AST types
    ast_assert_stmt = "AssertStmt"
    ast_explicit_constructor_invocation_stmt = "ExplicitConstructorInvocationStmt"
    ast_expression_stmt = "ExpressionStmt"
    ast_return_stmt = "ReturnStmt"
    ast_throw_stmt = "ThrowStmt"
    ast_local_class_declaration_stmt = "LocalClassDeclarationStmt"
    ast_local_record_declaration_stmt = "LocalRecordDeclarationStmt"
    ast_break_stmt = "BreakStmt"
    ast_continue_stmt = "ContinueStmt"
    ast_yield_stmt = "YieldStmt"
    # fmt: off
    asts_terminal_stmt = {
        ast_assert_stmt, ast_explicit_constructor_invocation_stmt, ast_expression_stmt,
        ast_return_stmt, ast_throw_stmt, 
        ast_local_class_declaration_stmt, ast_local_record_declaration_stmt,
        ast_break_stmt, ast_continue_stmt, ast_yield_stmt,
    }
    # fmt: on
    ast_block_stmt = "BlockStmt"
    ast_do_stmt = "DoStmt"
    ast_empty_stmt = "EmptyStmt"
    ast_for_each_stmt = "ForEachStmt"
    ast_for_stmt = "ForStmt"
    ast_if_stmt = "IfStmt"
    ast_labeled_stmt = "LabeledStmt"
    ast_switch_stmt = "SwitchStmt"
    ast_synchronized_stmt = "SynchronizedStmt"
    ast_try_stmt = "TryStmt"
    ast_while_stmt = "WhileStmt"
    ast_unparsable_stmt = "UnparsableStmt"
    # fmt: off
    asts_stmt = {
        ast_assert_stmt, ast_explicit_constructor_invocation_stmt, ast_expression_stmt,
        ast_return_stmt, ast_throw_stmt, 
        ast_local_class_declaration_stmt, ast_local_record_declaration_stmt,
        ast_break_stmt, ast_continue_stmt, ast_yield_stmt,
        ast_block_stmt, ast_do_stmt, ast_empty_stmt, ast_for_each_stmt, ast_for_stmt,
        ast_if_stmt, ast_labeled_stmt, ast_switch_stmt, ast_synchronized_stmt,
        ast_try_stmt, ast_while_stmt,
        ast_unparsable_stmt,
    }
    # fmt: on

    ast_annotation_declaration = "AnnotationDeclaration"
    ast_annotation_member_declaration = "AnnotationMemberDeclaration"
    ast_class_or_interface_declaration = "ClassOrInterfaceDeclaration"
    ast_compact_constructor_declaration = "CompactConstructorDeclaration"
    ast_constructor_declaration = "ConstructorDeclaration"
    ast_enum_constant_declaration = "EnumConstantDeclaration"
    ast_enum_declaration = "EnumDeclaration"
    ast_field_declaration = "FieldDeclaration"
    ast_initializer_declaration = "InitializerDeclaration"
    ast_method_declaration = "MethodDeclaration"
    ast_parameter = "Parameter"
    ast_receiver_parameter = "ReceiverParameter"
    ast_record_declaration = "RecordDeclaration"
    ast_variable_declarator = "VariableDeclarator"
    ast_array_access_expr = "ArrayAccessExpr"
    ast_array_creation_expr = "ArrayCreationExpr"
    ast_array_initializer_expr = "ArrayInitializerExpr"
    ast_assign_expr = "AssignExpr"
    ast_binary_expr = "BinaryExpr"
    ast_boolean_literal_expr = "BooleanLiteralExpr"
    ast_cast_expr = "CastExpr"
    ast_char_literal_expr = "CharLiteralExpr"
    ast_class_expr = "ClassExpr"
    ast_conditional_expr = "ConditionalExpr"
    ast_double_literal_expr = "DoubleLiteralExpr"
    ast_enclosed_expr = "EnclosedExpr"
    ast_field_access_expr = "FieldAccessExpr"
    ast_instance_of_expr = "InstanceOfExpr"
    ast_integer_literal_expr = "IntegerLiteralExpr"
    ast_lambda_expr = "LambdaExpr"
    ast_long_literal_expr = "LongLiteralExpr"
    ast_marker_annotation_expr = "MarkerAnnotationExpr"
    ast_member_value_pair = "MemberValuePair"
    ast_method_call_expr = "MethodCallExpr"
    ast_method_reference_expr = "MethodReferenceExpr"
    ast_name = "Name"
    ast_name_expr = "NameExpr"
    ast_normal_annotation_expr = "NormalAnnotationExpr"
    ast_null_literal_expr = "NullLiteralExpr"
    ast_object_creation_expr = "ObjectCreationExpr"
    ast_pattern_expr = "PatternExpr"
    ast_simple_name = "SimpleName"
    ast_single_member_annotation_expr = "SingleMemberAnnotationExpr"
    ast_string_literal_expr = "StringLiteralExpr"
    ast_super_expr = "SuperExpr"
    ast_switch_expr = "SwitchExpr"
    ast_text_block_literal_expr = "TextBlockLiteralExpr"
    ast_this_expr = "ThisExpr"
    ast_type_expr = "TypeExpr"
    ast_unary_expr = "UnaryExpr"
    ast_variable_declaration_expr = "VariableDeclarationExpr"
    ast_module_declaration = "ModuleDeclaration"
    ast_module_exports_directive = "ModuleExportsDirective"
    ast_module_opens_directive = "ModuleOpensDirective"
    ast_module_provides_directive = "ModuleProvidesDirective"
    ast_module_requires_directive = "ModuleRequiresDirective"
    ast_module_uses_directive = "ModuleUsesDirective"
    ast_catch_clause = "CatchClause"
    ast_switch_entry = "SwitchEntry"
    ast_array_type = "ArrayType"
    ast_class_or_interface_type = "ClassOrInterfaceType"
    ast_intersection_type = "IntersectionType"
    ast_primitive_type = "PrimitiveType"
    ast_type_parameter = "TypeParameter"
    ast_union_type = "UnionType"
    ast_unknown_type = "UnknownType"
    ast_var_type = "VarType"
    ast_void_type = "VoidType"
    ast_wildcard_type = "WildcardType"
    ast_array_creation_level = "ArrayCreationLevel"
    ast_compilation_unit = "CompilationUnit"
    ast_import_declaration = "ImportDeclaration"
    ast_modifier = "Modifier"
    ast_package_declaration = "PackageDeclaration"
    ast_javadoc_comment = "JavadocComment"
    ast_terminal = "Terminal"

    # fmt: off
    asts_all = {
        ast_assert_stmt, ast_explicit_constructor_invocation_stmt, ast_expression_stmt,
        ast_return_stmt, ast_throw_stmt,
        ast_local_class_declaration_stmt, ast_local_record_declaration_stmt,
        ast_break_stmt, ast_continue_stmt, ast_yield_stmt,
        ast_block_stmt, ast_do_stmt, ast_empty_stmt, ast_for_each_stmt, ast_for_stmt,
        ast_if_stmt, ast_labeled_stmt, ast_switch_stmt, ast_synchronized_stmt,
        ast_try_stmt, ast_while_stmt, ast_unparsable_stmt,
        ast_annotation_declaration, ast_annotation_member_declaration, ast_class_or_interface_declaration, ast_compact_constructor_declaration,
        ast_constructor_declaration, ast_enum_constant_declaration, ast_enum_declaration, ast_field_declaration, ast_initializer_declaration,
        ast_method_declaration, ast_parameter, ast_receiver_parameter, ast_record_declaration, ast_variable_declarator, ast_array_access_expr,
        ast_array_creation_expr, ast_array_initializer_expr, ast_assign_expr, ast_binary_expr, ast_boolean_literal_expr, ast_cast_expr, ast_char_literal_expr,
        ast_class_expr, ast_conditional_expr, ast_double_literal_expr, ast_enclosed_expr, ast_field_access_expr, ast_instance_of_expr, ast_integer_literal_expr, ast_lambda_expr, ast_long_literal_expr,
        ast_marker_annotation_expr, ast_member_value_pair, ast_method_call_expr, ast_method_reference_expr, ast_name, ast_name_expr, ast_normal_annotation_expr,
        ast_null_literal_expr, ast_object_creation_expr, ast_pattern_expr, ast_simple_name, ast_single_member_annotation_expr, ast_string_literal_expr, ast_super_expr, ast_switch_expr, ast_text_block_literal_expr, ast_this_expr,
        ast_type_expr, ast_unary_expr, ast_variable_declaration_expr, ast_module_declaration, ast_module_exports_directive, ast_module_opens_directive, ast_module_provides_directive,
        ast_module_requires_directive, ast_module_uses_directive, ast_catch_clause, ast_switch_entry, ast_array_type, ast_class_or_interface_type, ast_intersection_type, ast_primitive_type, ast_type_parameter, ast_union_type,
        ast_unknown_type, ast_var_type, ast_void_type, ast_wildcard_type, ast_array_creation_level, ast_compilation_unit, ast_import_declaration, ast_modifier, ast_package_declaration,
        ast_javadoc_comment, ast_terminal,
    }
    # fmt: on

    # token kinds
    tok_whitespace_no_eol = "WHITESPACE_NO_EOL"
    tok_eol = "EOL"
    tok_comment = "COMMENT"
    tok_identifier = "IDENTIFIER"
    tok_keyword = "KEYWORD"
    tok_literal = "LITERAL"
    tok_separator = "SEPARATOR"
    tok_operator = "OPERATOR"
    # fmt: off
    toks_all = {
        tok_whitespace_no_eol, tok_eol, tok_comment, tok_identifier, tok_keyword, tok_literal, tok_separator, tok_operator
    }
    # fmt: on

    # ===== bytecode =====

    # access opcode
    acc_public = 0x0001  # class, field, method
    acc_private = 0x0002  # class, field, method
    acc_protected = 0x0004  # class, field, method
    acc_static = 0x0008  # field, method
    acc_final = 0x0010  # class, field, method, parameter
    acc_super = 0x0020  # class
    acc_synchronized = 0x0020  # method
    acc_open = 0x0020  # module
    acc_transitive = 0x0020  # module requires
    acc_volatile = 0x0040  # field
    acc_bridge = 0x0040  # method
    acc_static_phase = 0x0040  # module requires
    acc_varargs = 0x0080  # method
    acc_transient = 0x0080  # field
    acc_native = 0x0100  # method
    acc_interface = 0x0200  # class
    acc_abstract = 0x0400  # class, method
    acc_strict = 0x0800  # method
    acc_synthetic = 0x1000  # class, field, method, parameter, module *
    acc_annotation = 0x2000  # class
    acc_enum = 0x4000  # class(?) field inner
    acc_mandated = 0x8000  # field, method, parameter, module, module *
    acc_module = 0x8000  # class

    # operator names
    op_aconst_null = "ACONST_NULL"
    op_iconst_m1 = "ICONST_M1"
    op_iconst_0 = "ICONST_0"
    op_iconst_1 = "ICONST_1"
    op_iconst_2 = "ICONST_2"
    op_iconst_3 = "ICONST_3"
    op_iconst_4 = "ICONST_4"
    op_iconst_5 = "ICONST_5"
    op_lconst_0 = "LCONST_0"
    op_lconst_1 = "LCONST_1"
    op_fconst_0 = "FCONST_0"
    op_fconst_1 = "FCONST_1"
    op_fconst_2 = "FCONST_2"
    op_dconst_0 = "DCONST_0"
    op_dconst_1 = "DCONST_1"

    op_bipush = "BIPUSH"
    op_sipush = "SIPUSH"
    op_ldc = "LDC"
    op_ldc_end = "LDC_END"

    op_iload = "ILOAD"
    op_lload = "LLOAD"
    op_fload = "FLOAD"
    op_dload = "DLOAD"
    op_aload = "ALOAD"
    ops_load_insn = {op_iload, op_lload, op_fload, op_dload, op_aload}

    op_iaload = "IALOAD"
    op_laload = "LALOAD"
    op_faload = "FALOAD"
    op_daload = "DALOAD"
    op_aaload = "AALOAD"
    op_baload = "BALOAD"
    op_caload = "CALOAD"
    op_saload = "SALOAD"
    # fmt: off
    ops_load_array_insn = {op_iaload, op_laload, op_faload, op_daload, op_aaload, op_baload, op_caload, op_saload}

    op_istore = "ISTORE"
    op_lstore = "LSTORE"
    op_fstore = "FSTORE"
    op_dstore = "DSTORE"
    op_astore = "ASTORE"
    ops_store_insn = {op_istore, op_lstore, op_fstore, op_dstore, op_astore}

    op_iastore = "IASTORE"
    op_lastore = "LASTORE"
    op_fastore = "FASTORE"
    op_dastore = "DASTORE"
    op_aastore = "AASTORE"
    op_bastore = "BASTORE"
    op_castore = "CASTORE"
    op_sastore = "SASTORE"
    #fmt: off
    ops_array_store_insn = {op_iastore, op_lastore, op_fastore, op_dastore, op_aastore, op_bastore, op_castore, op_sastore}

    op_pop = "POP"
    op_pop2 = "POP2"
    op_dup = "DUP"
    op_dup_x1 = "DUP_X1"
    op_dup_x2 = "DUP_X2"
    op_dup2 = "DUP2"
    op_dup2_x1 = "DUP2_X1"
    op_dup2_x2 = "DUP2_X2"
    op_swap = "SWAP"

    op_iadd = "IADD"
    op_ladd = "LADD"
    op_fadd = "FADD"
    op_dadd = "DADD"
    op_isub = "ISUB"
    op_lsub = "LSUB"
    op_fsub = "FSUB"
    op_dsub = "DSUB"
    op_imul = "IMUL"
    op_lmul = "LMUL"
    op_fmul = "FMUL"
    op_dmul = "DMUL"
    op_idiv = "IDIV"
    op_ldiv = "LDIV"
    op_fdiv = "FDIV"
    op_ddiv = "DDIV"
    op_irem = "IREM"
    op_lrem = "LREM"
    op_frem = "FREM"
    op_drem = "DREM"
    op_ineg = "INEG"
    op_lneg = "LNEG"
    op_fneg = "FNEG"
    op_dneg = "DNEG"

    op_ishl = "ISHL"
    op_lshl = "LSHL"
    op_ishr = "ISHR"
    op_lshr = "LSHR"
    op_iushr = "IUSHR"
    op_lushr = "LUSHR"
    op_iand = "IAND"
    op_land = "LAND"
    op_ior = "IOR"
    op_lor = "LOR"
    op_ixor = "IXOR"
    op_lxor = "LXOR"

    op_iinc = "IINC"

    op_i2l = "I2L"
    op_i2f = "I2F"
    op_i2d = "I2D"
    op_l2i = "L2I"
    op_l2f = "L2F"
    op_l2d = "L2D"
    op_f2i = "F2I"
    op_f2l = "F2L"
    op_f2d = "F2D"
    op_d2i = "D2I"
    op_d2l = "D2L"
    op_d2f = "D2F"
    op_i2b = "I2B"
    op_i2c = "I2C"
    op_i2s = "I2S"

    op_lcmp = "LCMP"
    op_fcmpl = "FCMPL"
    op_fcmpg = "FCMPG"
    op_dcmpl = "DCMPL"
    op_dcmpg = "DCMPG"

    op_ifeq = "IFEQ"
    op_ifne = "IFNE"
    op_iflt = "IFLT"
    op_ifge = "IFGE"
    op_ifgt = "IFGT"
    op_ifle = "IFLE"
    op_if_icmpeq = "IF_ICMPEQ"
    op_if_icmpne = "IF_ICMPNE"
    op_if_icmplt = "IF_ICMPLT"
    op_if_icmpge = "IF_ICMPGE"
    op_if_icmpgt = "IF_ICMPGT"
    op_if_icmple = "IF_ICMPLE"
    op_if_acmpeq = "IF_ACMPEQ"
    op_if_acmpne = "IF_ACMPNE"
    op_goto = "GOTO"

    op_tableswitch = "TABLESWITCH"
    op_lookupswitch = "LOOKUPSWITCH"

    op_ireturn = "IRETURN"
    op_lreturn = "LRETURN"
    op_freturn = "FRETURN"
    op_dreturn = "DRETURN"
    op_areturn = "ARETURN"
    op_return = "RETURN"

    op_getstatic = "GETSTATIC"
    op_putstatic = "PUTSTATIC"
    op_getfield = "GETFIELD"
    op_putfield = "PUTFIELD"
    ops_field_insn = {op_getstatic, op_putstatic, op_getfield, op_putfield}

    op_invokevirtual = "INVOKEVIRTUAL"
    op_invokespecial = "INVOKESPECIAL"
    op_invokestatic = "INVOKESTATIC"
    op_invokeinterface = "INVOKEINTERFACE"
    ops_method_insn = {
        op_invokevirtual,
        op_invokespecial,
        op_invokestatic,
        op_invokeinterface,
    }

    op_invokedynamic = "INVOKEDYNAMIC"
    op_lambda = "LAMBDA"
    op_lambda_end = "LAMBDA_END"

    op_new = "NEW"
    op_newarray = "NEWARRAY"
    op_anewarray = "ANEWARRAY"
    op_arraylength = "ARRAYLENGTH"
    op_athrow = "ATHROW"
    op_checkcast = "CHECKCAST"
    op_instanceof = "INSTANCEOF"
    op_multianewarray = "MULTIANEWARRAY"
    op_ifnull = "IFNULL"
    op_ifnonnull = "IFNONNULL"

    op_label = "LABEL"

    # fmt: off
    ops_all = {
        op_aconst_null, op_iconst_m1, op_iconst_0, op_iconst_1, op_iconst_2, 
        op_iconst_3, op_iconst_4, op_iconst_5, op_lconst_0, op_lconst_1, 
        op_fconst_0, op_fconst_1, op_fconst_2, op_dconst_0, op_dconst_1,
        op_bipush, op_sipush, op_ldc,
        op_iload, op_lload, op_fload, op_dload, op_aload,
        op_iaload, op_laload, op_faload, op_daload, op_aaload, 
        op_baload, op_caload, op_saload,
        op_istore, op_lstore, op_fstore, op_dstore, op_astore,
        op_iastore, op_lastore, op_fastore, op_dastore, op_aastore, 
        op_bastore, op_castore, op_sastore,
        op_pop, op_pop2, op_dup, op_dup_x1, op_dup_x2, 
        op_dup2, op_dup2_x1, op_dup2_x2, op_swap,
        op_iadd, op_ladd, op_fadd, op_dadd,
        op_isub, op_lsub, op_fsub, op_dsub,
        op_imul, op_lmul, op_fmul, op_dmul,
        op_idiv, op_ldiv, op_fdiv, op_ddiv,
        op_irem, op_lrem, op_frem, op_drem,
        op_ineg, op_lneg, op_fneg, op_dneg,
        op_ishl, op_lshl, op_ishr, op_lshr, op_iushr, op_lushr,
        op_iand, op_land, op_ior, op_lor, op_ixor, op_lxor,
        op_iinc,
        op_i2l, op_i2f, op_i2d, op_l2i, op_l2f, op_l2d,
        op_f2i, op_f2l, op_f2d, op_d2i, op_d2l, op_d2f,
        op_i2b, op_i2c, op_i2s,
        op_lcmp, op_fcmpl, op_fcmpg, op_dcmpl, op_dcmpg,
        op_ifeq, op_ifne, op_iflt, op_ifge, op_ifgt, op_ifle,
        op_if_icmpeq, op_if_icmpne, op_if_icmplt, op_if_icmpge, 
        op_if_icmpgt, op_if_icmple, op_if_acmpeq, op_if_acmpne,
        op_goto, op_tableswitch, op_lookupswitch,
        op_ireturn, op_lreturn, op_freturn, op_dreturn, op_areturn, op_return,
        op_getstatic, op_putstatic, op_getfield, op_putfield,
        op_invokevirtual, op_invokespecial, op_invokestatic, op_invokeinterface,
        op_invokedynamic,
        op_new, op_newarray, op_anewarray, op_arraylength, 
        op_athrow, op_checkcast, op_instanceof, 
        op_multianewarray, op_ifnull, op_ifnonnull,
        op_label,
    }
    # fmt: on


class TraverseOrder(enum.Enum):
    PRE_ORDER = "pre_order"
    DEPTH_FIRST = "dfs"


@dataclasses.dataclass
class AST:
    ast_type: str = ""
    tok_kind: Optional[str] = None
    tok: Optional[str] = None
    children: Optional[List["AST"]] = None
    lineno: str = None

    @classmethod
    def deserialize(cls, data: list) -> "AST":
        ast = cls()
        type_and_kind = data[0]
        ast.lineno = data[1]
        if ":" in type_and_kind:
            # terminal
            ast.ast_type, ast.tok_kind = type_and_kind.split(":")
            ast.tok = data[2]
        else:
            # non-terminal
            ast.ast_type = type_and_kind
            ast.children = [cls.deserialize(child) for child in data[2:]]
        return ast

    def serialize(self) -> list:
        if self.tok_kind is not None:
            # terminal
            return [f"{self.ast_type}:{self.tok_kind}", self.lineno, self.tok]
        else:
            # non-terminal
            return [self.ast_type, self.lineno] + [
                child.serialize() for child in self.children
            ]

    def __str__(self, indent: int = 0):
        s = self.ast_type
        if self.tok is not None:
            s += f"|{self.tok_kind}[{self.tok}]"
        if self.lineno is not None:
            s += f" <{self.lineno}>"
        if self.children is not None:
            s += " (\n" + " " * indent
            for child in self.children:
                s += "  " + child.__str__(indent + 2) + "\n" + " " * indent
            s += ")"
        return s

    def get_lineno_range(self) -> Tuple[int, int]:
        if self.lineno is None:
            raise RuntimeError("AST has no line number")
        if "-" in self.lineno:
            start, end = self.lineno.split("-")
            return int(start), int(end)
        else:
            return int(self.lineno), int(self.lineno)

    def is_terminal(self) -> bool:
        return self.children is None

    def traverse(
        self,
        stop_func: Callable[["AST"], bool] = lambda x: False,
        order: TraverseOrder = TraverseOrder.DEPTH_FIRST,
    ) -> Iterable["AST"]:
        """Traverse over each node in pre-order"""
        queue = [self]
        while len(queue) > 0:
            cur = queue.pop(0)
            yield cur

            if not stop_func(cur):
                if cur.children is not None:
                    if order == TraverseOrder.PRE_ORDER:
                        queue += cur.children
                    elif order == TraverseOrder.DEPTH_FIRST:
                        queue = cur.children + queue
                    else:
                        raise ValueError(f"Unknown traverse order: {order}")

    def get_tokens(self) -> List[str]:
        tokens = []
        for n in self.traverse():
            if n.is_terminal():
                tokens.append(n.tok)
        return tokens

    # def get_insns(self) -> List[Insn]:
    #     if self.insns is None:
    #         return []
    #     return [Insn(insn) for insn in self.insns]

    def get_body(self) -> "AST":
        """Extract the method body of this method declaration, which is always a BlockStmt node"""
        # find first BlockStmt in child
        block_stmt = None
        for child in self.children:
            if child.ast_type == "BlockStmt":
                block_stmt = child

        if block_stmt is None:
            raise RuntimeError("Method has no body")

        return block_stmt

    def get_sign(self) -> "AST":
        """Extract the sub-tree for the method signature (excluding method body, opening and closing parens)"""
        copy_node = copy.copy(self)
        copy_node.children = []

        for child in self.children:
            # stop at first block statement
            if child.ast_type == "BlockStmt":
                break
            copy_node.children.append(child)
        return copy_node

    def size(self, count_terminal: bool = True, count_nonterminal: bool = True) -> int:
        c = 0
        if self.is_terminal():
            if count_terminal:
                c += 1
        else:
            if count_nonterminal:
                c += 1
            for child in self.children:
                c += child.size(count_terminal, count_nonterminal)
        return c


@dataclasses.dataclass
class Insn:
    op: str = ""
    operands: List[str] = dataclasses.field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.op, *self.operands)

    @classmethod
    def deserialize(cls, data: List[str]) -> "Insn":
        insn = cls()
        insn.op = data[0]
        insn.operands = data[1:]
        return insn

    def serialize(self) -> List[str]:
        return [self.op] + self.operands

    def is_method_insn(self) -> bool:
        """check whether the instruction is a method invocation (not including invokedynamic)"""
        return self.op in Consts.ops_method_insn

    def is_field_insn(self) -> bool:
        """check whether the instruction is a field access"""
        return self.op in Consts.ops_field_insn

    def get_owner(self) -> str:
        """get the owner class name of this method/field invocation"""
        if self.is_method_insn() or self.is_field_insn():
            return self.operands[0]
        else:
            raise RuntimeError("Not a method or field invocation")

    def get_method_namedesc(self) -> str:
        """get the method name+desc of this method invocation"""
        if self.is_method_insn():
            # name + desc
            return self.operands[1] + self.operands[2]
        else:
            raise RuntimeError("Not a method invocation")

    def get_method_name(self) -> str:
        """get the method name of this method invocation"""
        if self.is_method_insn():
            return self.operands[1]
        else:
            raise RuntimeError("Not a method invocation")

    def get_method_ptypes(self) -> List[str]:
        """get the method parameter types of this method invocation or invokedynamic"""
        if self.is_method_insn():
            ptypes = self.operands[2].split(")")[0].split("(")[1]
            # TODO: cannot split by "," any more
            return [] if len(ptypes) == 0 else self.desc_split_types(ptypes)
        elif self.op == Consts.op_invokedynamic:
            if self.operands[0] == Consts.op_lambda:
                ptypes = self.operands[1].split(")")[0].split("(")[1]
                return [] if len(ptypes) == 0 else self.desc_split_types(ptypes)
            elif self.operands[0] in Consts.ops_method_insn:
                pytpes = self.operands[3].split(")")[0].split("(")[1]
                return [] if len(pytpes) == 0 else self.desc_split_types(pytpes)
            else:
                raise RuntimeError("Not a supported invokedynamic type")
        else:
            raise RuntimeError("Not a method invocation or invokedynamic")

    def get_field_name(self) -> str:
        """get the field name of this field access"""
        if self.is_field_insn():
            return self.operands[1]
        else:
            raise RuntimeError("Not a field access")

    def get_class_canonical_name(self, i: int) -> str:
        """get the class canonical name at operand #i"""
        return self.operands[i].split(".")[-1]

    def is_jump_insn(self) -> bool:
        """check whether the instruction is a jump instruction"""
        return self.op in {
            "IFEQ",
            "IFNE",
            "IFLT",
            "IFGE",
            "IFGT",
            "IFLE",
            "IF_ICMPEQ",
            "IF_ICMPNE",
            "IF_ICMPLT",
            "IF_ICMPGE",
            "IF_ICMPGT",
            "IF_ICMPLE",
            "IF_ACMPEQ",
            "IF_ACMPNE",
            "GOTO",
            "IFNULL",
            "IFNONNULL",
        }

    def is_label(self) -> bool:
        return self.op == "LABEL"

    def get_label(self, i: int = 0) -> int:
        """get the jump target at operand #i (default #0 for all jump instructions and label)"""
        return int(self.operands[i])

    def __str__(self):
        return self.op + " " + " ".join(self.operands)

    def get_tokens(self) -> List[str]:
        return [self.op] + self.operands

    @classmethod
    def convert_to_tokens(self, insns: List["Insn"]) -> List[str]:
        """
        converts a list of instructions to a list of tokens (separated by spaces).
        """
        tokens = []
        for insn in insns:
            tokens += insn.get_tokens()
        return tokens

    @classmethod
    def convert_from_tokens(self, tokens: List[str]) -> List["Insn"]:
        """
        converts a list of tokens back to list of instructions.
        """
        insns = []
        cur_insn = []
        for token in tokens:
            if token in Consts.ops_all and len(cur_insn) > 0:
                if cur_insn[0] in Consts.ops_method_insn and len(cur_insn) in {1, 2}:
                    # skip any identifier that coincident with op names in method insn
                    pass
                elif cur_insn[0] in Consts.ops_field_insn and len(cur_insn) in {1, 2}:
                    # skip any identifier that coincident with op names in field insn
                    pass
                elif cur_insn[0] == Consts.op_ldc and cur_insn[-1] != Consts.op_ldc_end:
                    # skip any identifier that coincident with op names in unfinished ldc
                    pass
                else:
                    # start a new instruction
                    insns.append(Insn.deserialize(cur_insn))
                    cur_insn = []

            cur_insn.append(token)
        if len(cur_insn) > 0:
            insns.append(Insn.deserialize(cur_insn))
        return insns

    @classmethod
    def class_q2iname(self, qname: str) -> str:
        return qname.replace(".", "/")

    @classmethod
    def class_i2qname(self, iname: str) -> str:
        return iname.replace("/", ".")

    @classmethod
    def class_name2desc(self, name: str) -> str:
        if name.endswith("[]"):
            return "[" + self.class_name2desc(name[:-2])

        if name == "int":
            return "I"
        elif name == "long":
            return "J"
        elif name == "float":
            return "F"
        elif name == "double":
            return "D"
        elif name == "boolean":
            return "Z"
        elif name == "byte":
            return "B"
        elif name == "char":
            return "C"
        elif name == "short":
            return "S"
        elif name == "void":
            return "V"
        else:
            return "L" + name.replace(".", "/") + ";"

    @classmethod
    def desc_split_types(self, desc: str) -> List[str]:
        """
        split the descriptor into a list of types.
        """
        types = []
        last_split_point = 0
        i = 0

        def split():
            nonlocal desc, types, last_split_point, i
            types.append(desc[last_split_point : i + 1])
            last_split_point = i + 1

        while i < len(desc):
            c = desc[i]
            if c == "[":
                i += 1
                continue
            elif c == "L":
                i += 1
                while desc[i] != ";":
                    i += 1
                split()
            else:
                split()
            i += 1

        return types


class SimpInsnType(enum.Enum):
    const = "<C>"
    method = "<M>"
    get_field = "<gF>"
    put_field = "<pF>"
    get_local = "<gL>"
    put_local = "<pL>"
    decl = "<decl>"  # local variable or class declaration
    ret = "<ret>"  # return statement
    throw = "<throw>"  # throw statement
    multi = "<multi>"  # multiple assignments in one statement
    invalid = "<INVALID>"


@dataclasses.dataclass
class SimpInsnNode:
    # the type of this plan
    type: SimpInsnType = None
    # the local variable index, or field/method name
    name: Optional[str] = None
    # arguments
    args: Optional[List["SimpInsnNode"]] = None
    # number of arguments
    cargs: int = 0
    # extra info: int constant value, or field/method id
    extra: Optional[int] = None

    def pre_order(self) -> List[str]:
        ret = [self.type.value]
        if self.name is not None:
            ret.append(self.name)
        if self.args is not None:
            for arg in self.args:
                ret.extend(arg.pre_order())
        return ret

    def pre_order_nodes(self) -> List["SimpInsnNode"]:
        ret = [self]
        if self.args is not None:
            for arg in self.args:
                ret.extend(arg.pre_order_nodes())
        return ret

    def post_order(self) -> List[str]:
        ret = []
        if self.args is not None:
            for arg in self.args:
                ret.extend(arg.post_order())
        ret.append(self.type.value)
        if self.name is not None:
            ret.append(self.name)
        return ret

    def height(self) -> int:
        if self.args is None:
            return 1
        else:
            return 1 + max([arg.height() for arg in self.args])

    def __str__(self) -> str:
        s = self.type.value
        if self.name is not None:
            s += " " + self.name
        if self.extra is not None:
            s += f"[{self.extra}]"
        if self.args is not None:
            s += " ("
            for arg in self.args:
                s += " " + arg.__str__() + ","
            s += " )"
        return s

    @classmethod
    def deserialize(cls, data) -> "SimpInsnNode":
        field_values = {}
        for f in dataclasses.fields(cls):
            if f.name in data:
                if f.name == "args":
                    ftype = List[SimpInsnNode]
                else:
                    ftype = f.type
                field_values[f.name] = su.io.deserialize(data.get(f.name), ftype)
        return cls(**field_values)

    def get_all_mids(self) -> List[int]:
        """get all method ids used in this plan, excluding INVOKEDYNAMIC"""
        mids = []
        for node in self.pre_order_nodes():
            if node.type == SimpInsnType.method:
                if node.extra is not None:
                    mids.append(node.extra)
                elif node.name != "INVOKEDYNAMIC":
                    logger.warning(f"method {node.name} was not resolved")
        return mids

    def get_all_fids(self) -> List[int]:
        """get all field ids used in this plan"""
        fids = []
        for node in self.pre_order_nodes():
            if (
                node.type == SimpInsnType.get_field
                or node.type == SimpInsnType.put_field
            ):
                if node.extra is not None:
                    fids.append(node.extra)
                else:
                    logger.warning(f"field {node.name} was not resolved")
        return fids

    def equals_modulo_const(self, other: "SimpInsnNode") -> bool:
        if not isinstance(other, SimpInsnNode):
            return False

        type_name_cargs_match = all(
            [
                self.type == other.type,
                self.name == other.name,
                self.cargs == other.cargs,
            ]
        )

        if not type_name_cargs_match:
            return False

        if self.args is not None and other.args is not None:
            args_match = True
            if len(self.args) != len(other.args):
                args_match = False
            else:
                for i in range(len(self.args)):
                    if not self.args[i].equals_modulo_const(other.args[i]):
                        args_match = False
                        break
        elif self.args is None and other.args is None:
            args_match = True
        else:
            args_match = False

        if not args_match:
            return False

        if self.type == SimpInsnType.const:
            return True
        else:
            return self.extra == other.extra


@dataclasses.dataclass
class MethodBytecode:
    insns: Dict[int, Insn] = dataclasses.field(default_factory=dict)
    lnt: Dict[int, int] = dataclasses.field(default_factory=dict)
    # TODO: extract names of local variables

    @classmethod
    def deserialize(cls, data: dict) -> "MethodBytecode":
        return cls(
            insns={
                int(k): v
                for k, v in su.io.deserialize(data["insns"], Dict[int, Insn]).items()
            },
            lnt={
                int(k): v
                for k, v in su.io.deserialize(data["lnt"], Dict[int, int]).items()
            },
        )

    def get_ordered_insns(self) -> List[Insn]:
        """get instructions in order"""
        return [self.insns[i] for i in sorted(self.insns.keys())]


@dataclasses.dataclass
class ClassStructure:
    id: int = -1
    access: int = 0
    scope: Scope = Scope.APP
    ext: int = -1
    impl: List[int] = dataclasses.field(default_factory=list)
    name: str = ""
    fields: List[int] = dataclasses.field(default_factory=list)
    methods: List[int] = dataclasses.field(default_factory=list)

    def is_public(self) -> bool:
        return (self.access & Consts.acc_public) != 0

    def is_private(self) -> bool:
        return (self.access & Consts.acc_private) != 0

    def is_protected(self) -> bool:
        return (self.access & Consts.acc_protected) != 0

    def is_package(self) -> bool:
        return (
            (self.access & Consts.acc_public) == 0
            and (self.access & Consts.acc_private) == 0
            and (self.access & Consts.acc_protected == 0)
        )

    def is_interface(self) -> bool:
        return (self.access & Consts.acc_interface) != 0

    @property
    def iname(self) -> str:
        return Insn.class_q2iname(self.name)

    @property
    def simple_name(self) -> str:
        if len(self.name) == 0:
            return ""
        return self.name.split(".")[-1]

    def __hash__(self) -> int:
        return self.id


@dataclasses.dataclass
class MethodStructure:
    id: int = -1
    access: int = 0
    clz: int = -1
    is_test: bool = False
    name: str = ""
    ptypes: List[str] = dataclasses.field(default_factory=list)
    rtype: str = ""
    ttypes: List[str] = dataclasses.field(default_factory=list)
    atypes: List[str] = dataclasses.field(default_factory=list)
    code: Optional[str] = None
    ast: Optional[AST] = None
    bytecode: Optional[MethodBytecode] = None

    def is_public(self) -> bool:
        return (self.access & Consts.acc_public) != 0

    def is_private(self) -> bool:
        return (self.access & Consts.acc_private) != 0

    def is_protected(self) -> bool:
        return (self.access & Consts.acc_protected) != 0

    def is_package(self) -> bool:
        return (
            (self.access & Consts.acc_public) == 0
            and (self.access & Consts.acc_private) == 0
            and (self.access & Consts.acc_protected == 0)
        )

    def is_static(self) -> bool:
        return (self.access & Consts.acc_static) != 0

    def is_abstract(self) -> bool:
        return (self.access & Consts.acc_abstract) != 0

    @property
    def sign(self) -> str:
        return self.name + "(" + ",".join(self.ptypes) + ")" + self.rtype

    @property
    def desc(self) -> str:
        return (
            "("
            + "".join([Insn.class_name2desc(t) for t in self.ptypes])
            + ")"
            + Insn.class_name2desc(self.rtype)
        )

    @property
    def namedesc(self) -> str:
        return self.name + self.desc

    def get_num_args(self) -> int:
        if self.is_static():
            return len(self.ptypes)
        else:
            return len(self.ptypes) + 1

    def __hash__(self) -> int:
        return self.id


@dataclasses.dataclass
class FieldStructure:
    id: int = -1
    access: int = 0
    clz: int = -1
    name: str = ""
    type: str = ""

    def is_public(self) -> bool:
        return (self.access & Consts.acc_public) != 0

    def is_private(self) -> bool:
        return (self.access & Consts.acc_private) != 0

    def is_protected(self) -> bool:
        return (self.access & Consts.acc_protected) != 0

    def is_package(self) -> bool:
        return (
            (self.access & Consts.acc_public) == 0
            and (self.access & Consts.acc_private) == 0
            and (self.access & Consts.acc_protected == 0)
        )

    def is_static(self) -> bool:
        return (self.access & Consts.acc_static) != 0

    def __hash__(self) -> int:
        return self.id


@dataclasses.dataclass
class TestResult:
    suite: str = ""
    classname: Optional[str] = None
    name: str = ""
    time: float = 0
    result: str = "pass"
    msg: Optional[str] = None  # error/failure/skipped message
    etype: Optional[str] = None  # error/failure type


PRIMITIVE_INAME2SNAME = {
    "I": "int",
    "Z": "boolean",
    "B": "byte",
    "C": "char",
    "S": "short",
    "F": "float",
    "J": "long",
    "D": "double",
}


def simplify_type_name(t: str) -> str:
    # take off the array part
    array_level = 0
    while t.startswith("["):
        array_level += 1
        t = t[1:]

    if t in PRIMITIVE_INAME2SNAME:
        # it is a primitive type's internal name
        t = PRIMITIVE_INAME2SNAME[t]
    else:
        # if ends with ";", take off it and the initial "L"
        if t.endswith(";"):
            t = t[1:-1]

        # split by "." and "/" and only keep the last part
        t = t.split(".")[-1]
        t = t.split("/")[-1]

    # add the array part back (in simple name way)
    for i in range(array_level):
        t += "[]"
    return t
