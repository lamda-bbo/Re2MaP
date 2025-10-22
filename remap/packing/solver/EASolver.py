import numpy as np
from numpy import ndarray

import random
import logging
from typing import Generic, TypeVar, List, Tuple, Callable

from .competition import TournamentCompetition
from .competition import FitnessSoftmaxProbabilityCompetition as DefaultCompetition
from ..common.definition import Genotype, Verifier, Evaluator, Scorer, Competition
from ...common.definition import Factory

logger = logging.getLogger("EASolver")

T = TypeVar("T", bound=Genotype)

class EASolver(Generic[T]):
    _scorer: Scorer
    _verifier: Verifier
    _evaluator: Evaluator
    
    _competition: Competition
    
    def __init__(self, *, factory: Factory[T], num_pops=5, num_offs=5):
        self._factory = factory
        self.num_pops = num_pops
        self.num_offs = num_offs
        
        self._scorer = None
        self._verifier = None
        self._evaluator = None
        
        self._competition = TournamentCompetition(2)
        
    @property
    def scorer(self) -> Scorer:
        return self._scorer
    
    @scorer.setter
    def scorer(self, _scorer: Scorer):
        self._scorer = _scorer
        
    def _score(self, metric_list):
        return self._scorer(metric_list)    
    
    @property
    def verifier(self) -> Verifier:
        return self._verifier
    
    @verifier.setter
    def verifier(self, _verifier: Verifier):
        self._verifier = _verifier
    
    def _verify(self, pop) -> bool:
        return self._verifier(pop)
    
    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator
    
    @evaluator.setter
    def evaluator(self, _evaluator: Evaluator):
        self._evaluator = _evaluator
        
    def _evaluate(self, pop):
        return self._evaluator(pop)
        
    @property
    def competition(self):
        return self._competition
        
    @competition.setter
    def competition(self, _competition: Competition):
        self._competition = _competition
        
    def _compete(self, population, metrics, fitness, *, round=1) -> List[T]:
        return self._competition(population, metrics, fitness, round=round)
        
    def _populate(self, is_guaranteed_valid=False) -> T:
        pop = self._factory()
        if is_guaranteed_valid and self._verifier is not None:
            while not self._verify(pop):
                pop = self._factory()
        return pop
    
    def _mutate(self, gen: T, op=None) -> T:
        cls = type(gen)
        op = op or (cls.mutate if cls.mutable else None)
        return op(gen) if op else gen

    def _crossover(self, gen1: T, gen2: T, op=None) -> T:
        cls = type(gen1)
        op = op or (cls.crossover if cls.crossable else None)
        return op(gen1, gen2) if op else random.choice([gen1, gen2])
    
    def __call__(self, num_evaluation=1000, *, verbose=True, log_interval=None) -> Tuple[List[T], List, ndarray, ndarray]:
        if verbose:
            log_interval = log_interval or (num_evaluation // 10)
            def log_interval_handler():
                next_interval = log_interval
                def reach_interval(num_evals):
                    nonlocal next_interval
                    if num_evals >= next_interval:
                        next_interval = min(next_interval + log_interval, num_evaluation)
                        return True
                    return False
                while True:
                    yield reach_interval
            log_hdlr = log_interval_handler()
        
        population = [self._populate() for _ in range(self.num_pops)]
        gtype = type(population[0])
        metrics = [self._evaluate(pop) for pop in population]
        fitness = np.array(self._score(metrics))
        if self._verifier:
            valid = np.array([self._verify(pop) for pop in population])
            vmasked = lambda fitness, valid: np.where(valid, fitness, fitness + np.max(fitness) + 1e5)
            fitness = vmasked(fitness, valid)
            
        num_evals = self.num_pops
        
        trace = []
        def _update_trace(num_evals, fitness):
            nonlocal trace
            trace.append((num_evals, np.min(fitness)))
            
        def logfmt(num_evals, metrics, fitness):
            f = lambda s, *args, **kwargs: s.format(*args, **kwargs)
            pascal_case: Callable[[str], str] = \
                lambda s: "".join([item.capitalize() for item in s.split("_")])
            content = ", ".join([
                f("# evaluation {:4d}", num_evals) if num_evals is not None else "final",
                f("BestScore {:.6E}", fitness[0]),
                ", ".join([f("{} {:.6E}", pascal_case(key), value) for key, value in metrics[0].items()]),
            ])
            return content
            
        while num_evals < num_evaluation:
            offsprings: List[T] = []
            for _ in range(min(self.num_offs, num_evaluation - num_evals)):
                if gtype.crossable:
                    parents_id = self._compete(population, metrics, fitness, round=2)
                    parents: List[T] = [population[parent_id] for parent_id in parents_id]
                    offsprings.append(self._mutate(self._crossover(*parents)))
                else:
                    prototype = population[self._compete(population, metrics, fitness, round=1)[0]]
                    offsprings.append(self._mutate(prototype))
        
            _population = population + offsprings
            _metrics = metrics + [self._evaluate(off) for off in offsprings]
            _fitness = np.array(self._score(_metrics))
            if self._verifier:
                _valid = np.concatenate([valid, np.array([self._verify(off) for off in offsprings])])
                _fitness = vmasked(_fitness, _valid)
            num_evals += len(offsprings)
            
            next_generation = np.argsort(_fitness)[:self.num_pops]
            population = [_population[index] for index in next_generation]
            metrics = [_metrics[index] for index in next_generation]
            fitness = np.array(self._score(metrics))
            if self._verifier:
                valid = np.array([_valid[index] for index in next_generation])
                fitness = vmasked(fitness, valid)
            _update_trace(num_evals, fitness)
            
            if verbose and next(log_hdlr)(num_evals):
                logger.info(logfmt(num_evals, metrics, fitness))
                
        if verbose:
            logger.info(logfmt(None, metrics, fitness))
        return population, metrics, fitness, np.array(trace)