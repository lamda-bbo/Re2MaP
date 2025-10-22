import numpy as np
from numpy import ndarray

from typing import List, Tuple, Type, Generic, TypeVar

from .sequence import Sequence
from .packing_tree import PackingTree, BBTSPackingTree, NBTPackingTree
from ..common.definition import Genotype
from ..common.contour import Contour
from ...common.definition import Factory

PT_TV = TypeVar('PT_TV', bound=PackingTree)

class Anchor:
    def __init__(self, x, bottom, left):
        self.x, self.bottom, self.left = \
             x,      bottom,      left

class MPTree(Generic[PT_TV], Genotype["MPTree"]):
    PT: Type[PT_TV]
    SQ = Sequence
    
    def __init__(self, anchors: List[Anchor], factories: List[Factory[PackingTree]], *, initialize=True):
        assert len(anchors) == len(factories)
        self.num_trees = len(anchors)
        self.anchors = anchors
        self.factories = factories
        self.pts = None
        self.seq = None
        if initialize:
            self.initialize()
            
    def initialize(self):
        self.pts = [f() for f in self.factories]
        self.seq = type(self).SQ(list(range(self.num_trees)))
    
    @classmethod
    def crossover(cls, gen1, gen2):
        assert gen1.num_trees == gen2.num_trees
        assert np.all(np.array([f1 is f2 for f1, f2 in zip(gen1.factories, gen2.factories)])), (gen1.factories, gen2.factories)
        newgen = cls(gen1.anchors, gen1.factories, initialize=True)
        newgen.pts = [cls.PT.crossover(pt1, pt2) if pt1 is not None and pt2 is not None else None
                      for pt1, pt2 in zip(gen1.pts, gen2.pts)]
        newgen.seq = cls.SQ.crossover(gen1.seq, gen2.seq)
        return newgen

    @classmethod
    def mutate(cls, gen):
        newgen = cls(gen.anchors, gen.factories, initialize=False)
        newgen.pts = [cls.PT.mutate(pt) if pt is not None else None for pt in gen.pts]
        newgen.seq = cls.SQ.mutate(gen.seq)
        return newgen
    
    def topl(self, block_w, block_h, xl, xh, yl, yh) -> Tuple[ndarray, ndarray]:
        ctb, ctt = Contour(), Contour()
        ctb.initialize(xl, xh, platform=yl, bottom=True)
        ctt.initialize(xl, xh, platform=yh, bottom=False)
        x, y = np.full_like(block_w, np.nan), np.full_like(block_h, np.nan)
        w, h = block_w, block_h
        for ptId in self.seq.gene:
            ac: Anchor = self.anchors[ptId]
            pt: PackingTree | None = self.pts[ptId]
            if pt is None:
                continue
            _x, _y = pt.topl(
                w[pt.blocks], h[pt.blocks],
                ac.x, ac.bottom, ac.left,
                contour=ctb if ac.bottom else ctt
            )
            x[pt.blocks] = _x
            y[pt.blocks] = _y
        return x, y
            
class MPTreeMetaClass(Generic[PT_TV]):
    @staticmethod
    def make(packing_tree: Type[PT_TV]) -> Type[MPTree[PT_TV]]:
        class ConcreteMPTree(MPTree):
            PT = packing_tree
            mutable = packing_tree.mutable
            crossable = packing_tree.crossable
            
        return ConcreteMPTree

class BBTSMPTree(MPTreeMetaClass.make(BBTSPackingTree)):
    """ multi-packing-trees implemented by combining bit-binary-tree and sequence """
    
class NBTMPTree(MPTreeMetaClass.make(NBTPackingTree)):
    """ multi-packing-trees based on node-binary-tree """