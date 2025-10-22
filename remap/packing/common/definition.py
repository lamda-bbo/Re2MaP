from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Generic, TypeVar, Type, List, Dict
from numbers import Real

from ...common.definition import FlyweightPool

G = TypeVar('G', bound="Genotype")
T = TypeVar('T')
V = TypeVar('V')

## Genotype

class Genotype(Generic[G], ABC):
    mutable = False
    crossable = False
    
    @classmethod
    @abstractmethod
    def mutate(cls: Type[G], gen: G) -> G: ...

    @classmethod
    @abstractmethod
    def crossover(cls: Type[G], gen1: G, gen2: G) -> G: ...

def is_mutable(cls: Genotype):
    return cls.mutable

def is_crossable(cls: Genotype):
    return cls.crossable

## Verifier, Evaluator and Scorer

class Verifier(Generic[G], ABC):
    @abstractmethod
    def __call__(self, gen: G) -> bool: ...
    
class Evaluator(Generic[G], ABC):
    @abstractmethod
    def __call__(self, gen: G) -> Dict[str, Real]: ...
    
class Scorer(ABC):
    @abstractmethod
    def __call__(self, metrics: List[Dict[str, Real]]) -> List[Real]: ...
    
## Competition

class Competition(Generic[G], ABC):
    @abstractmethod
    def __call__(self, population: List[G], metrics: List[dict], fitness: ndarray, *, round=1) -> List[G]: ...
    
## EvalOp

class EvalOp(ABC):
    name: str
    
    def __init__(self, name: str = None):
        self.name = name or type(self).name
    
    @staticmethod
    @abstractmethod
    def evaluate(pool: FlyweightPool) -> Real: ...
    
    def __call__(self, pool: FlyweightPool) -> Dict[str, Real]:
        return {self.name: self.evaluate(pool)}