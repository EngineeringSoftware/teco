import collections
import functools
import sys
from pathlib import Path
from typing import Dict, List

import seutil as su

from teco.data.cg import CallGraph
from teco.data.pesudo_structures import PesudoStructures
from teco.data.structures import (
    ClassStructure,
    Consts,
    FieldStructure,
    Insn,
    MethodStructure,
)

logger = su.log.get_logger(__name__)


class ClassResolutionException(Exception):
    pass


class MethodResolutionException(Exception):
    pass


class FieldResolutionException(Exception):
    pass


class RawDataLoader:
    def __init__(self, indexed: bool = False):
        # increase recursion limit to be able to load some crazy code with long BinaryExpr chain
        sys.setrecursionlimit(10000)

        self.indexed = indexed

        # currently loaded project
        self.loaded_proj_data_dir = None

        # project raw data space
        self.classes: List[ClassStructure] = []
        self.methods: List[MethodStructure] = []
        self.fields: List[FieldStructure] = []
        self.cg: CallGraph = None

        # persistent JRE data during collection; they always take up the lower indices
        self.classes_jre: List[ClassStructure] = []
        self.methods_jre: List[MethodStructure] = []
        self.fields_jre: List[FieldStructure] = []

        # extra raw data space for pesudo structures; they start from a very high index
        self.classes_extra: List[ClassStructure] = PesudoStructures.get_classes()
        self.methods_extra: List[MethodStructure] = PesudoStructures.get_methods()
        self.fields_extra: List[FieldStructure] = PesudoStructures.get_fields()

        if self.indexed:
            # class, method, and field indexes
            self.iname2cid: Dict[str, ClassStructure] = {}
            self.cid2namedesc2mid: Dict[int, Dict[str, int]] = collections.defaultdict(
                dict
            )
            self.cid2name2fid: Dict[int, Dict[str, int]] = collections.defaultdict(dict)

    def load_jre_data(self, jre_data_dir: Path):
        self.classes_jre = su.io.load(
            jre_data_dir / "joint.class.json", clz=List[ClassStructure]
        )
        self.methods_jre = su.io.load(
            jre_data_dir / "joint.method.json", clz=List[MethodStructure]
        )
        self.fields_jre = su.io.load(
            jre_data_dir / "joint.field.json", clz=List[FieldStructure]
        )

    def get_class(self, cid: int) -> ClassStructure:
        if cid >= PesudoStructures.reserved_cid_begin:
            return self.classes_extra[cid - PesudoStructures.reserved_cid_begin]
        else:
            return self.classes[cid]

    @property
    def all_classes(self) -> List[ClassStructure]:
        return self.classes + self.classes_extra

    def get_method(self, mid: int) -> MethodStructure:
        if mid >= PesudoStructures.reserved_mid_begin:
            return self.methods_extra[mid - PesudoStructures.reserved_mid_begin]
        else:
            return self.methods[mid]

    @property
    def all_methods(self) -> List[MethodStructure]:
        return self.methods + self.methods_extra

    def get_field(self, fid: int) -> FieldStructure:
        if fid >= PesudoStructures.reserved_fid_begin:
            return self.fields_extra[fid - PesudoStructures.reserved_fid_begin]
        else:
            return self.fields[fid]

    @property
    def all_fields(self) -> List[FieldStructure]:
        return self.fields + self.fields_extra

    def load_project_data(self, proj_data_dir: Path) -> bool:
        if self.loaded_proj_data_dir == proj_data_dir:
            return False

        self.classes.clear()
        self.classes += self.classes_jre
        self.methods.clear()
        self.methods += self.methods_jre
        self.fields.clear()
        self.fields += self.fields_jre

        # load project data
        self.classes += su.io.load(
            proj_data_dir / "joint.class.json", clz=List[ClassStructure]
        )
        if len(self.classes) >= PesudoStructures.reserved_cid_begin:
            raise RuntimeError(f"Too many classes: {len(self.classes)}")
        self.methods += su.io.load(
            proj_data_dir / "joint.method.json", clz=List[MethodStructure]
        )
        if len(self.methods) >= PesudoStructures.reserved_mid_begin:
            raise RuntimeError(f"Too many methods: {len(self.methods)}")
        self.fields += su.io.load(
            proj_data_dir / "joint.field.json", clz=List[FieldStructure]
        )
        if len(self.fields) >= PesudoStructures.reserved_fid_begin:
            raise RuntimeError(f"Too many fields: {len(self.fields)}")

        self.cg = su.io.load(proj_data_dir / "joint.cg.json", clz=CallGraph)

        if self.indexed:
            # invalid lookup cache
            self.lookup_virtual_method.cache_clear()
            self.lookup_interface_method.cache_clear()
            self.lookup_special_method.cache_clear()
            self.lookup_static_method.cache_clear()
            self.lookup_field.cache_clear()

            self.iname2cid = {
                Insn.class_q2iname(cs.name): cs.id for cs in self.all_classes
            }
            self.cid2namedesc2mid = collections.defaultdict(dict)
            self.cid2name2fid = collections.defaultdict(dict)
            for cid in self.iname2cid.values():
                cs = self.get_class(cid)
                for mid in cs.methods:
                    self.cid2namedesc2mid[cs.id][self.get_method(mid).namedesc] = mid
                for fid in cs.fields:
                    self.cid2name2fid[cs.id][self.get_field(fid).name] = fid

        self.loaded_proj_data_dir = proj_data_dir
        return True

    def lookup_class(self, name: str) -> int:
        try:
            return self.iname2cid[name]
        except KeyError:
            raise ClassResolutionException("ClassNotFoundError")

    def lookup_method(self, cid: int, namedesc: str, op: str) -> int:
        if op == Consts.op_invokevirtual:
            return self.lookup_virtual_method(cid, namedesc)
        elif op == Consts.op_invokespecial:
            return self.lookup_special_method(cid, namedesc)
        elif op == Consts.op_invokestatic:
            return self.lookup_static_method(cid, namedesc)
        elif op == Consts.op_invokeinterface:
            return self.lookup_interface_method(cid, namedesc)
        else:
            raise MethodResolutionException(f"Unknown op: {op}")

    @functools.lru_cache(maxsize=10240)
    def lookup_virtual_method(self, cid: int, namedesc: str) -> int:
        """
        https://docs.oracle.com/javase/specs/jvms/se8/jvms8.pdf section 5.4.3.3
        """
        cs = self.get_class(cid)

        # 1. if C is an interface, method resolution throws an IncompatibleClassChangeError.
        if cs.is_interface():
            raise MethodResolutionException("IncompatibleClassChangeError")

        # 2.  Otherwise, method resolution attempts to locate the referenced method in C and its superclasses:

        # 2.1. [IGNORED] If C declares exactly one method with the name specified by the method reference, and the declaration is a signature polymorphic method (§2.9), then method lookup succeeds. All the class names mentioned in the descriptor are resolved (§5.4.3.1).

        # 2.2. Otherwise, if C declares a method with the name and descriptor specified by the method reference, method lookup succeeds.
        mid = self.cid2namedesc2mid[cid].get(namedesc, -2)

        # 2.3. Otherwise, if C has a superclass, step 2 of method resolution is recursively invoked on the direct superclass of C
        if mid < 0:
            super_cid = cs.ext
            while super_cid > 0:
                super_cs = self.get_class(super_cid)
                mid = self.cid2namedesc2mid.get(super_cid, {}).get(namedesc, -2)
                if mid >= 0:
                    break
                super_cid = super_cs.ext

        # 3. Otherwise, method resolution attempts to locate the referenced method in the superinterfaces of the specified class C:
        if mid < 0:
            # 3.1 If the maximally-specific superinterface methods of C for the name and descriptor specified by the method reference include exactly one method that does not have its ACC_ABSTRACT flag set, then this method is chosen and method lookup succeeds.
            # 3.2 Otherwise, if any superinterface of C declares a method with the name and descriptor specified by the method reference that has neither its ACC_PRIVATE flag nor its ACC_STATIC flag set, one of these is arbitrarily chosen and method lookup succeeds.
            interface_queue = []
            interface_visited = set()
            potential_mids = []

            # perform a breath-first search to find all superinterfaces
            interface_queue += cs.impl
            while len(interface_queue) > 0:
                interface_cid = interface_queue.pop(0)
                if interface_cid in interface_visited or interface_cid < 0:
                    continue
                interface_visited.add(interface_cid)

                potential_mid = self.cid2namedesc2mid.get(interface_cid, {}).get(
                    namedesc, -2
                )
                if potential_mid >= 0:
                    potential_ms = self.get_method(potential_mid)
                    if potential_ms.is_private() or potential_ms.is_static():
                        pass
                    if potential_ms.is_abstract():
                        potential_mids.append(potential_mid)
                    else:
                        # found a non-abstract method in the maximally-specific superinterface
                        mid = potential_mid
                        break

                interface_queue += self.get_class(interface_cid).impl

            if len(potential_mids) > 0:
                # randomly choose one
                mid = potential_mids[0]

        if mid < 0:
            raise MethodResolutionException("NoSuchMethodError")

        return mid

    @functools.lru_cache(maxsize=10240)
    def lookup_interface_method(self, cid: int, namedesc: str) -> int:
        """
        https://docs.oracle.com/javase/specs/jvms/se8/jvms8.pdf section 5.4.3.4
        """
        cs = self.get_class(cid)

        # 1. If C is not an interface, interface method resolution throws an IncompatibleClassChangeError.
        if not cs.is_interface():
            raise MethodResolutionException("IncompatibleClassChangeError")

        # 2. Otherwise, if C declares a method with the name and descriptor specified by the interface method reference, method lookup succeeds.
        mid = self.cid2namedesc2mid[cid].get(namedesc, -2)

        # 3. Otherwise, if the class Object declares a method with the name and descriptor specified by the interface method reference, which has its ACC_PUBLIC flag set and does not have its ACC_STATIC flag set, method lookup succeeds.
        if mid < 0:
            object_cid = self.iname2cid["java/lang/Object"]
            potential_mid = self.cid2namedesc2mid[object_cid].get(namedesc, -2)
            if potential_mid >= 0:
                potential_ms = self.get_method(potential_mid)
                if not potential_ms.is_static() and potential_ms.is_public():
                    mid = potential_mid

        # 4. Otherwise, if the maximally-specific superinterface methods (§5.4.3.3) of C for the name and descriptor specified by the method reference include exactly one method that does not have its ACC_ABSTRACT flag set, then this method is chosen and method lookup succeeds.
        # 5. Otherwise, if any superinterface of C declares a method with the name and descriptor specified by the method reference that has neither its ACC_PRIVATE flag nor its ACC_STATIC flag set, one of these is arbitrarily chosen and method lookup succeeds.
        if mid < 0:
            interface_queue = []
            interface_visited = set()
            potential_mids = []

            # perform a breath-first search to find all superinterfaces
            interface_queue += cs.impl
            while len(interface_queue) > 0:
                interface_cid = interface_queue.pop(0)
                if interface_cid in interface_visited or interface_cid < 0:
                    continue
                interface_visited.add(interface_cid)

                potential_mid = self.cid2namedesc2mid.get(interface_cid, {}).get(
                    namedesc, -2
                )
                if potential_mid >= 0:
                    potential_ms = self.get_method(potential_mid)
                    if potential_ms.is_private() or potential_ms.is_static():
                        pass
                    if potential_ms.is_abstract():
                        potential_mids.append(potential_mid)
                    else:
                        # found a non-abstract method in the maximally-specific superinterface
                        mid = potential_mid
                        break

                interface_queue += self.get_class(interface_cid).impl

            if len(potential_mids) > 0:
                # randomly choose one
                mid = potential_mids[0]

        if mid < 0:
            raise MethodResolutionException("NoSuchMethodError")

        return mid

    @functools.lru_cache(maxsize=10240)
    def lookup_special_method(self, cid: int, namedesc: str) -> int:
        """
        first try virtual, then try interface
        """
        try:
            mid = self.lookup_virtual_method(cid, namedesc)
        except MethodResolutionException:
            mid = self.lookup_interface_method(cid, namedesc)
        return mid

    @functools.lru_cache(maxsize=10240)
    def lookup_static_method(self, cid: int, namedesc: str) -> int:
        """
        find the exact method in cid, no overriding
        """
        mid = self.cid2namedesc2mid[cid].get(namedesc, -2)
        if mid < 0:
            raise MethodResolutionException("NoSuchMethodError")
        return mid

    @functools.lru_cache(maxsize=10240)
    def lookup_field(self, cid: int, name: str) -> int:
        """
        https://docs.oracle.com/javase/specs/jvms/se8/jvms8.pdf section 5.4.3.2
        """
        cs = self.get_class(cid)

        # 1. If C declares a field with the name and descriptor specified by the field reference, field lookup succeeds. The declared field is the result of the field lookup.
        fid = self.cid2name2fid[cid].get(name, -2)

        # 2. Otherwise, field lookup is applied recursively to the direct superinterfaces of the specified class or interface C.
        if fid < 0:
            for interface_cid in cs.impl:
                if interface_cid < 0:
                    continue
                fid = self.cid2name2fid[interface_cid].get(name, -2)
                if fid >= 0:
                    break

        # 3. Otherwise, if C has a superclass S, field lookup is applied recursively to S.
        if fid < 0:
            super_cid = cs.ext
            if super_cid >= 0:
                fid = self.lookup_field(super_cid, name)

        # 4. Otherwise, field lookup fails.
        if fid < 0:
            raise FieldResolutionException("NoSuchFieldError")

        return fid
