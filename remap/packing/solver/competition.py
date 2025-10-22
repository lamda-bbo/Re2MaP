import numpy as np
from ..common.definition import Competition

class FitnessSoftmaxProbabilityCompetition(Competition):
    def __init__(self, lowerbetter=True):
        self.lowerbetter = lowerbetter
        
    def __call__(self, population, metrics, fitness, *, round=1):
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
        asprobs = (lambda x: softmax(np.min(x) - x)) if self.lowerbetter else (lambda x: softmax(x - np.max(x)))
        winners = np.random.choice(np.arange(len(population), dtype=np.int_), size=round, p=asprobs(fitness))
        return winners

class TournamentCompetition(Competition):
    def __init__(self, num_competitors=2, *, lowerbetter=True):
        self.num_competitors = num_competitors
        self.lowerbetter = lowerbetter
    
    def __call__(self, population, metrics, fitness, *, round=1):
        winners = []
        for _ in range(round):
            competitors = np.random.choice(np.arange(len(population), dtype=np.int_), size=self.num_competitors)
            winner = competitors[(np.argmin if self.lowerbetter else np.argmax)(fitness[competitors])]
            winners.append(winner)
        return winners