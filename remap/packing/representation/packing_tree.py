import numpy as np
from numpy import ndarray

from abc import abstractmethod
from typing import Tuple, TypeVar

from .tree import BitBinaryTree, NodeBinaryTree
from .sequence import Sequence
from ..common.definition import Genotype
from ..common.contour import Contour

PT_TYPE = TypeVar("PT_TYPE", bound="PackingTree")
class PackingTree(Genotype[PT_TYPE]):
    def __init__(self, blocks, **kwargs):
        self.num_blocks = len(blocks)
        self.raw_blocks = blocks
        self.blocks = np.sort(np.array(blocks))
    
    @abstractmethod
    def topl(self, block_w: ndarray, block_h: ndarray, anchor_x, bottom: bool, left: bool, contour: Contour) -> Tuple[ndarray, ndarray]:...
    

class BitBinaryTreeSequencePackingTree(PackingTree["BitBinaryTreeSequencePackingTree"]):
    """ packing-tree implemented by combining bit-binary-tree and sequence """
    
    T = BitBinaryTree
    SQ = Sequence
    
    mutable = T.mutable
    crossable = T.crossable
    
    tree: T
    seq: SQ
    
    def __init__(self, blocks, *, initialize=True):
        super().__init__(blocks)
        self.tree = None
        self.seq = None
        if initialize:
            self.initialize()
            
    def initialize(self):
        self.tree = type(self).T(self.num_blocks)
        self.seq = type(self).SQ(self.blocks)
        
    @classmethod
    def crossover(cls, gen1, gen2):
        assert gen1.num_blocks == gen2.num_blocks
        assert np.all(gen1.blocks == gen2.blocks)
        newgen = cls(gen1.raw_blocks, initialize=False)
        newgen.tree = cls.T.crossover(gen1.tree, gen2.tree)
        newgen.seq = cls.SQ.crossover(gen1.seq, gen2.seq)
        return newgen
    
    @classmethod
    def mutate(cls, gen):
        newgen = cls(gen.raw_blocks, initialize=False)
        newgen.tree = cls.T.mutate(gen.tree)
        newgen.seq = cls.SQ.mutate(gen.seq)
        return newgen
    
    def topl(self, block_w, block_h, anchor_x, bottom, left, contour):
        binary_tree = lambda size, dtype: np.zeros(size, dtype=dtype)
        tree_size = self.tree.tree_size
        binary_tree_ts = lambda dtype: binary_tree(tree_size, dtype)
        xt = binary_tree_ts(np.float_)
        wt = binary_tree_ts(np.float_)
        ht = binary_tree_ts(np.float_)
        tt = binary_tree_ts(np.int_)
        pt = binary_tree_ts(np.int_)
        xt[0] = anchor_x if left else anchor_x - block_w[self.seq.gene[0]]
        tt[0] = 0
        wt[self.tree.gene] = block_w[self.seq.gene]
        ht[self.tree.gene] = block_h[self.seq.gene]
        _last = lambda cb: (cb + 1) // 2 - 1
        _next = lambda cb: (cb + 1) *  2 - 1
        cb = 1
        for h in range(1, self.tree.max_height):
            lb = _last(cb)
            nb = _next(cb)
            tt[cb:nb] = np.concatenate([tt[lb:cb] + 1, tt[lb:cb] + tt[cb - 1] + 1])
            pt[cb:nb] = tt[cb:nb] + np.arange(0, nb - cb) * (2 ** (self.tree.max_height - h) - 1)
            xt[cb+0:nb:2] = xt[lb:cb] if left else xt[lb:cb] + wt[lb:cb] - wt[cb+0:nb:2]
            xt[cb+1:nb:2] = xt[lb:cb] + wt[lb:cb] if left else xt[lb:cb] - wt[cb+1:nb:2]
            cb = _next(cb)
        
        ptvsl = np.argsort(pt[self.tree.gene])
        ptnds = self.tree.nodes[ptvsl]
        
        xl, width, height = xt[ptnds], wt[ptnds], ht[ptnds]
        yl = np.zeros_like(xl)
        for index, (x, w, h) in enumerate(zip(xl, width, height)):
            platform = contour.get_platform(x, x + w)
            yl[index] = platform if bottom else platform - h
            platform = platform + h if bottom else platform - h
            contour.add_segment(x, x + w, platform)
        
        x, y = np.zeros_like(xl), np.zeros_like(yl)
        x[self.seq.gene[ptvsl]] = xl
        y[self.seq.gene[ptvsl]] = yl
        return x, y

BBTSPackingTree = BitBinaryTreeSequencePackingTree

class NodeBinaryTreePackingTree(PackingTree["NodeBinaryTreePackingTree"]):
    T = NodeBinaryTree
    
    mutable = T.mutable
    crossable = T.crossable
    
    def __init__(self, blocks, *, initialize=True):
        super().__init__(blocks)
        self.tree = None
        if initialize:
            self.initialize()
            
    def initialize(self):
        self.tree = type(self).T(self.blocks)
    
    @classmethod
    def mutate(cls, gen):
        newgen = cls(gen.blocks, initialize=False)
        newgen.tree = cls.T.mutate(gen.tree)
        return newgen
    
    @classmethod
    def crossover(cls, gen1, gen2):
        raise NotImplementedError
    
    def topl(self, block_w, block_h, anchor_x, bottom, left, contour):
        xl = np.zeros(self.num_blocks, dtype=np.float_)
        yl = np.zeros(self.num_blocks, dtype=np.float_)
        def assign_y(x, w, h):
            platform = contour.get_platform(x, x + w)
            y = platform if bottom else platform - h
            platform = platform + h if bottom else platform - h
            contour.add_segment(x, x + w, platform)
            return y
        
        nodes, node2block = np.unique(self.tree.nodes, return_inverse=True)
        root = self.tree.root
        root_block = node2block[root]
        stack = [root]
        xl[root_block] = anchor_x if left else anchor_x - block_w[root_block]
        while len(stack) > 0:
            this = stack.pop()
            this_block = node2block[this]
            yl[this_block] = assign_y(xl[this_block], block_w[this_block], block_h[this_block])
            
            rchild = self.tree.rchild[this]
            if rchild != -1:
                child_block = node2block[rchild]
                xl[child_block] = xl[this_block] if left else xl[this_block] + block_w[this_block] - block_w[child_block]
                stack.append(rchild)
            
            lchild = self.tree.lchild[this]
            if lchild != -1:
                child_block = node2block[lchild]
                xl[child_block] = xl[this_block] + block_w[this_block] if left else xl[this_block] - block_w[child_block]
                stack.append(lchild)
                
        return xl, yl
    
NBTPackingTree = NodeBinaryTreePackingTree