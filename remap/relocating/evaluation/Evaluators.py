import numpy as np
from numpy import ndarray, intp
from typing import List, Type, Union, Callable

from .PropertyPools import PropertyPool
from ...common.definition import Factory
from ...packing.common.definition import Genotype, Evaluator, EvalOp
from ...common.timer import Timer

timer = Timer()

PoolFactory = Union[Type[PropertyPool], Callable[..., PropertyPool], Factory[PropertyPool]]

class Evaluators(Evaluator[Genotype]):
    """ multiple evaluators """
    def __init__(self, pf: PoolFactory, eval_ops: List[EvalOp]):
        self._eval_ops = eval_ops
        self._pf = pf
        
    def append(self, eval_op):
        self._eval_ops.append(eval_op)
    
    @timer.listen("evaluation")
    def __call__(self, gen):
        pool = self._pf(gen)
        metric = {}
        for eval_op in self._eval_ops:
            metric.update(eval_op(pool))
        return metric