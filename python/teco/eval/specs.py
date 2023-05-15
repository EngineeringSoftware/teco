import dataclasses
import functools
from typing import Dict, List, Set


@dataclasses.dataclass
class ModelSpec:
    trials: List[str] = dataclasses.field(default_factory=list)
    tags: Set[str] = dataclasses.field(default_factory=set)


ModelsSpec = Dict[str, ModelSpec]


@dataclasses.dataclass
class SetupsSpec:
    setup: List[str] = dataclasses.field(default_factory=list)
    eval_set: List[str] = dataclasses.field(default_factory=list)
    decoding: List[str] = dataclasses.field(default_factory=list)
    default_metrics: List[str] = dataclasses.field(default_factory=list)

    @functools.cached_property
    def total(self) -> int:
        return len(self.setup) * len(self.eval_set) * len(self.decoding)
