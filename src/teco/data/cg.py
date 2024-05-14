import dataclasses
from typing import Dict, Set, Tuple

import seutil as su


class EdgeType:
    CALL = "C"
    OVERRIDE = "O"


@dataclasses.dataclass
class CallGraph:
    edges: Dict[int, Set[Tuple[int, str]]] = dataclasses.field(default_factory=dict)

    @classmethod
    def deserialize(cls, data) -> "CallGraph":
        cg = cls()
        cg.edges = {int(k): v for k, v in su.io.deserialize(data["edges"]).items()}
        return cg

    def get_edges_from(self, mid: int) -> Set[Tuple[int, str]]:
        """
        Get all edges starting from the node (specified by its key).

        Time complexity: O(log(V))
        """
        return self.edges.get(mid, set())

    def get_edges_to(self, mid: int) -> Set[Tuple[int, str]]:
        """
        Get all edges targeting to the node (specified by its key).

        Time complexity: O(E)
        """
        collected_edges: Set[Tuple[int, str]] = set()
        for from_key, to_labels in self.edges.items():
            for to_key, label in to_labels:
                if to_key == mid:
                    collected_edges.add((from_key, label))

        return collected_edges

    def add_edge(self, from_mid: int, to_mid: int, edge_type: EdgeType = EdgeType.CALL):
        self.edges.setdefault(from_mid, set()).add((to_mid, edge_type))
