import numpy as np
from numpy import ndarray

from ..common.definition import Genotype

class Sequence(Genotype["Sequence"]):
    mutable = True
    crossable = True
    
    def __init__(self, nodes):
        self.num_nodes = len(nodes)
        self.raw_nodes = nodes
        self.nodes = np.sort(np.array(nodes))
        self.gene: ndarray | None = None
        self.initialize()
        
    def initialize(self):
        self.gene = np.arange(self.num_nodes, dtype=int)
        np.random.shuffle(self.gene)
        
    @classmethod
    def crossover(cls, gen1, gen2):
        assert gen1.num_nodes == gen2.num_nodes
        assert np.all(gen1.nodes == gen2.nodes)
        if np.random.randint(2) > 0:
            gen1, gen2 = gen2, gen1
        newgen = cls(gen1.raw_nodes)
        newgen.gene = gen2.gene[np.argsort(gen1.gene)]
        return newgen
    
    @classmethod
    def mutate(cls, gen):
        smask = np.random.randint(0, 5 * gen.num_nodes, size=gen.num_nodes) < 5
        sbits = np.where(smask)[0]
        np.random.shuffle(sbits)
        newgen = cls(gen.raw_nodes)
        newgen.gene = gen.gene.copy()
        newgen.gene[smask] = newgen.gene[sbits]
        return newgen