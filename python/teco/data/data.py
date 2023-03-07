import dataclasses
import functools
from typing import Dict, List, Optional, Tuple

import seutil as su

from teco.data.structures import AST, simplify_type_name

logger = su.log.get_logger(__name__)


BASIC_FIELDS = [
    "test_sign",
    "test_stmts",
    "test_stmt_insns",
    "test_stmt_fqinsns",
    "focalm",
    "proj_name",
    "test_cid",
    "test_mid",
    "test_mkey",
    "focal_cid",
    "focal_mid",
    "focal_mkey",
]


@dataclasses.dataclass
class Data:
    id: str = "default"
    fully_deserialized: bool = True

    # --- basic fields ---

    # test signature (ast)
    test_sign: AST = None

    @functools.cached_property
    def test_sign_toks(self) -> List[str]:
        return self.test_sign.get_tokens()

    # test stmts (ast)
    test_stmts: List[AST] = None

    @functools.cached_property
    def test_stmt_toks(self) -> List[List[str]]:
        return [stmt.get_tokens() for stmt in self.test_stmts]

    # test stmts' bytecode instructions (tokens)
    test_stmt_insns: List[List[str]] = None

    # test stmts' bytecode instructions, with all fully qualified names kept (tokens)
    test_stmt_fqinsns: List[List[str]] = None

    # focal method (ast)
    focalm: AST = None

    @functools.cached_property
    def focalm_toks(self) -> List[str]:
        return self.focalm.get_tokens()

    # data source information
    proj_name: str = None
    test_cid: int = None
    test_mid: int = None
    test_mkey: str = None
    focal_cid: int = None
    focal_mid: int = None
    focal_mkey: str = None

    # --- extra fields ---

    # local variables' types before each statement + after the last statement
    types_local: List[List[str]] = None

    @functools.cached_property
    def types_local_simplified(self) -> List[List[str]]:
        # types_local normalized to simple type names
        if self.types_local is None:
            return None
        else:
            return [
                [simplify_type_name(t) for t in types] for types in self.types_local
            ]

    # types that are needed for invoking the focal method and are not available yet
    types_absent: List[List[str]] = None

    @functools.cached_property
    def types_absent_simplified(self) -> List[List[str]]:
        # types_absent normalized to simple type names
        if self.types_absent is None:
            return None
        else:
            return [
                [simplify_type_name(t) for t in types] for types in self.types_absent
            ]

    # list of fields that are set / not set in focal class
    fields_set: List[List[str]] = None
    fields_notset: List[List[str]] = None

    # setup and teardown methods (asts, mids, names)
    setup_methods: List[AST] = None
    setup_mids: List[int] = None
    teardown_methods: List[AST] = None
    teardown_mids: List[int] = None

    # the last called (APP/TEST) method in previous statements
    # "None" either means that no method was called in the particular previous statement (i.e., should look into more prior statements OR no method has been called yet)
    last_called_methods: List[Optional[AST]] = None

    # for each statement, the statement from APP code with the most similar prior context
    similar_stmts: List[Optional[Tuple[float, AST]]] = None

    # the runtime types and values of variables of interest at each statement, saved as delta (only changed variables)
    runtime_types_values_delta: List[Dict[str, Tuple[str, str]]] = None

    # the runtime types and values of variables of interest at each statement (all variables)
    @functools.cached_property
    def runtime_types_values_all(self) -> List[Dict[str, Tuple[str, str]]]:
        if self.runtime_types_values_delta is None:
            return None
        else:
            ret = []
            all_types_values = {}
            for delta_types_values in self.runtime_types_values_delta:
                all_types_values.update(delta_types_values)
                ret.append(all_types_values.copy())
            return ret

    def runtime_data_valid(self) -> bool:
        if (
            self.runtime_types_values_delta is None
            or len(self.runtime_types_values_delta) != len(self.test_stmts) + 1
        ):
            return False
        return True

    def resolve_last_called_method(self, stmt_i: int) -> Optional[AST]:
        """
        Resolve the last called method when predicting statement #stmt_i,
        considering the fact that "None" means no method call at last statement only.
        """
        while self.last_called_methods[stmt_i] is None:
            stmt_i -= 1
            if stmt_i < 0:
                return None
        return self.last_called_methods[stmt_i]

    def cutoff(self, stmt_i: int):
        """
        Cut off this data to only the initial few statements.
        """
        self.test_stmts = self.test_stmts[:stmt_i]
        self.test_stmt_insns = self.test_stmt_insns[:stmt_i]
        self.test_stmt_fqinsns = self.test_stmt_fqinsns[:stmt_i]

        if self.types_local is not None:
            self.types_local = self.types_local[: stmt_i + 1]
        if self.types_absent is not None:
            self.types_absent = self.types_absent[: stmt_i + 1]
        if self.fields_set is not None:
            self.fields_set = self.fields_set[: stmt_i + 1]
        if self.fields_notset is not None:
            self.fields_notset = self.fields_notset[: stmt_i + 1]
        if self.last_called_methods is not None:
            self.last_called_methods = self.last_called_methods[: stmt_i + 1]
        if self.similar_stmts is not None:
            self.similar_stmts = self.similar_stmts[: stmt_i + 1]
        if self.runtime_types_values_delta is not None:
            self.runtime_types_values_delta = self.runtime_types_values_delta[
                : stmt_i + 1
            ]

    def finish_deserialization(self):
        if self.fully_deserialized:
            return

        fields: Dict[str, dataclasses.Field] = {
            f.name: f for f in dataclasses.fields(Data)
        }
        del fields["id"]
        del fields["fully_deserialized"]

        for name, field in fields.items():
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, su.io.deserialize(v, clz=field.type))

        self.fully_deserialized = True
