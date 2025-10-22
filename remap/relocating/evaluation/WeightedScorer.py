import numpy as np
from numpy import float_
from collections import defaultdict

from ...packing.common.definition import Scorer

class WeightedScorer(Scorer):
    def __init__(self, *keepraw, history=True, default_weight=0., **weights):
        self.history = history
        self.weights = defaultdict(lambda: default_weight, **weights)
        self.keepraw = set(keepraw)
        self.minv = {}
        self.maxv = {}
    
    def __call__(self, metrics):
        scores = []
        if not self.history:
            self.minv, self.maxv = {}, {}
        minv, maxv = self.minv, self.maxv
        for metric in metrics:
            for key, value in metric.items():
                if key not in self.keepraw:
                    minv.setdefault(key, value)
                    minv[key] = min(minv[key], value)
                    maxv.setdefault(key, value)
                    maxv[key] = max(maxv[key], value)
                
        normalize = lambda key, value: (value - minv[key] + 1e-5) / (maxv[key] - minv[key] + 1e-5)
        for metric in metrics:
            weighted_scores = [(value if key in self.keepraw else normalize(key, value)) * self.weights[key] for key, value in metric.items()]
            score = np.sum(weighted_scores, dtype=float_)
            scores.append(score)
        scores = np.array(scores, dtype=float_)
        return scores