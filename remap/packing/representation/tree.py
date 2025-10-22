import numpy as np
from numpy import ndarray
import random
import math
from typing import TypeVar, Callable

from ..common.definition import Genotype

BT_TYPE = TypeVar('BT_TYPE', bound="BinaryTree")

class BinaryTree(Genotype[BT_TYPE]):
    """
    Represents a binary tree, i.e., a connected graph G=(V,E) satisfying:
    - Graph G contains |V|=N vertices (nodes) and exactly |E|=N-1 edges
    - Every node has degree at most 3
    """
    
Tree = BinaryTree

class BitBinaryTree(Tree["BitBinaryTree"]):
    """
    A space-efficient binary tree representation using bitwise encoding.
    
    Structure:
    - Represents trees with maximum height H using 2^H - 1 bits
    - Each bit corresponds to a node in a complete binary tree's level-order:
        * 1 = node exists
        * 0 = node absent
    - Root node always exists (first bit must be 1)

    Example (H=3, 7-bit encoding):
      Bitstream "1100100" represents:
          Level 0: [1]    (root exists)
          Level 1: [1][0] (left child exists, right absent)
          Level 2: [0][1][0][0] (only left grandchild exists)
    """
    
    mutable = True
    crossable = True
    # crossable = False
    
    def __init__(self, num_nodes, *, legalize=True):
        self.num_nodes = num_nodes
        self.gene = None
        self.initialize()
        if legalize:
            self.legalize()
            
    def initialize(self):
        self.max_height = math.ceil(2.25 * math.log(self.num_nodes, 2)) + 1
        self.tree_size = 2 ** self.max_height - 1
        self.gene = np.zeros(self.tree_size, dtype=np.bool_)
        self.gene[:self.num_nodes] = 1
        np.random.shuffle(self.gene)
        
    @property
    def nodes(self):
        return np.where(self.gene)[0]
    
    @staticmethod
    def get_father_nodes(nodes, *, unique=False):
        father_nodes = np.floor_divide(nodes + 1, 2) - 1
        father_nodes[father_nodes < 0] = 0
        return np.unique(father_nodes) if unique else father_nodes
    
    @staticmethod
    def get_brother_nodes(nodes):
        brother_nodes = np.where(nodes % 2 == 0, nodes - 1, nodes + 1)
        return brother_nodes
    
    @property
    def non_leaf_nodes(self):
        return self.get_father_nodes(self.nodes, unique=True)
    
    def is_valid(self):
        return np.all(self.gene[self.non_leaf_nodes]) and (self.num_nodes == len(self.nodes))
    
    def legalize(self):
        if len(self.nodes) < self.num_nodes:
            num_additional_nodes = self.num_nodes - len(self.nodes)
            nonexistent = np.where(~self.gene)[0]
            np.random.shuffle(nonexistent)
            self.gene[nonexistent[:num_additional_nodes]] = 1
        
        nodes = self.get_father_nodes(self.nodes, unique=True)
        while len(nodes) > 0 and not np.all(self.gene[nodes]):
            self.gene[nodes] = 1
            nodes = self.get_father_nodes(self.nodes, unique=True)
        
        num_exceeded_nodes = len(self.nodes)  - self.num_nodes
        assert num_exceeded_nodes >= 0, (self.show(), self.nodes, num_exceeded_nodes, self.num_nodes)[1:]
        if num_exceeded_nodes == 0:
            assert len(self.nodes) == self.num_nodes, (self.show(), len(self.nodes), self.num_nodes)[1:]
            return
        
        leaf_nodes_mask = self.gene.copy()
        leaf_nodes_mask[self.non_leaf_nodes] = 0
        leaf_nodes = np.where(leaf_nodes_mask)[0]
        
        assert leaf_nodes is not None and len(leaf_nodes) > 0, (self.show(), leaf_nodes)[1:]
        
        while num_exceeded_nodes > 0:
            num_chosen_nodes = np.random.randint(1, min(len(leaf_nodes), num_exceeded_nodes) + 1)
            np.random.shuffle(leaf_nodes)
            chosen_nodes = leaf_nodes[:num_chosen_nodes]
            leaf_nodes = leaf_nodes[num_chosen_nodes:]
            
            self.gene[chosen_nodes] = 0
            num_exceeded_nodes -= num_chosen_nodes
            
            assert num_exceeded_nodes >= 0, (self.show(), num_exceeded_nodes, self.num_nodes)[1:]
            
            if num_exceeded_nodes > 0:
                brother_nodes = self.get_brother_nodes(chosen_nodes)
                father_nodes = self.get_father_nodes(chosen_nodes)
                new_leaf_nodes = np.unique(father_nodes[~self.gene[brother_nodes]])
                leaf_nodes = np.concatenate([leaf_nodes, new_leaf_nodes])
            
        assert len(self.nodes) == self.num_nodes, (self.show(), len(self.nodes), self.num_nodes)[1:]
        
    @classmethod
    def crossover(cls, gen1, gen2):
        assert gen1.num_nodes == gen2.num_nodes
        tree_size = gen1.tree_size
        mask = np.random.randint(0, 2, tree_size, dtype=np.bool_)
        newgen = cls(gen1.num_nodes, legalize=False)
        newgen.gene = np.where(mask, gen1.gene, gen2.gene)
        if not newgen.is_valid:
            newgen.legalize()
        return newgen
    
    @classmethod
    def mutate(cls, gen):
        mask = np.random.randint(0, 100, gen.tree_size)
        newgen = cls(gen.num_nodes, legalize=False)
        newgen.gene = np.where(mask < 5, ~gen.gene, gen.gene)
        if not newgen.is_valid():
            newgen.legalize()
        return newgen
    
    def show(self):
        cb = 0
        _next = lambda cb: (cb + 1) * 2 - 1
        for h in range(self.max_height):
            nb = _next(cb)
            nodes = self.gene[cb:nb].astype(int)
            print(*tuple(nodes.tolist()), sep=" " * (2 ** (self.max_height - h) - 1))
            cb = _next(cb)
            
class NodeBinaryTree(Tree["NodeBinaryTree"]):
    mutable = True
    crossable = False
    
    nodes: ndarray
    fixed_mask: ndarray
    root: np.int_
    parent: ndarray
    lchild: ndarray
    rchild: ndarray
        
    def __init__(self, nodes, *, initialize=True):
        self.nodes = nodes
        self.fixed_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        if initialize:
            self.initialize()
            
    def fix(self, nodes):
        self.fixed_mask[np.isin(self.nodes, nodes)] = 1
        
    def unfix(self):
        self.fixed_mask[:] = 0
            
    @property
    def num_nodes(self):
        return len(self.nodes)
    
    @property
    def leaf_mask(self):
        return np.logical_and(self.lchild == -1, self.rchild == -1)
            
    def initialize(self):
        self.parent = np.full(self.num_nodes, -1, dtype=np.int_)
        self.lchild = np.full(self.num_nodes, -1, dtype=np.int_)
        self.rchild = np.full(self.num_nodes, -1, dtype=np.int_)
        symmetrize: Callable[[ndarray], ndarray] = lambda X: (X - X.T) / 2
        probs = symmetrize(np.random.rand(self.num_nodes, self.num_nodes))
        np.fill_diagonal(probs, 0)
        self.root = np.argmax(np.sum(probs, axis=1))
        connected = np.zeros(self.num_nodes, dtype=np.bool_)
        connected[self.root] = 1
        exposed = np.zeros(self.num_nodes, dtype=np.bool_)
        exposed[self.root] = 1
        while np.any(~connected):
            exposed_nodes = np.where(exposed)[0]
            disconnected_nodes = np.where(~connected)[0]
            _probs = probs[exposed_nodes, :][:, disconnected_nodes]
            row, col = np.unravel_index(np.argmax(np.abs(_probs)), _probs.shape)
            prob = _probs[row, col]
            parent = exposed_nodes[row]
            child = disconnected_nodes[col]
            self.parent[child] = parent
            if self.lchild[parent] == -1 and self.rchild[parent] == -1:
                if prob < 0:
                    self.lchild[parent] = child
                else:
                    self.rchild[parent] = child
            else:
                if self.lchild[parent] == -1:
                    self.lchild[parent] = child
                else:
                    self.rchild[parent] = child
                exposed[parent] = 0
            connected[child] = 1
            exposed[child] = 1

    @classmethod
    def mutate(cls, gen, *, mutate_prob=2/3):
        newgen = cls(gen.nodes, initialize=False)
        newgen.fixed_mask = gen.fixed_mask.copy()
        newgen.root = gen.root
        newgen.parent = gen.parent.copy()
        newgen.lchild = gen.lchild.copy()
        newgen.rchild = gen.rchild.copy()
        targets = np.where(np.logical_and(~gen.leaf_mask, ~gen.fixed_mask))[0]
        if len(targets) == 0:
            return newgen
        
        def swap(g: NodeBinaryTree, x):
            g.lchild[x], g.rchild[x] = \
                g.rchild[x], g.lchild[x]

        def rotate_l(g: NodeBinaryTree, x):
            y = g.rchild[x]
            z = g.parent[x]
            if z != -1:
                if g.lchild[z] == x:
                    g.lchild[z] = y
                else:
                    g.rchild[z] = y
            else:
                g.root = y
            g.parent[x], g.parent[y] = y, z
            g.rchild[x], g.lchild[y] = g.lchild[y], x
            if g.rchild[x] != -1:
                g.parent[g.rchild[x]] = x
        
        def rotate_r(g: NodeBinaryTree, x):
            y = g.lchild[x]
            z = g.parent[x]
            if z != -1:
                if g.lchild[z] == x:
                    g.lchild[z] = y
                else:
                    g.rchild[z] = y
            else:
                g.root = y
            g.parent[x], g.parent[y] = y, z
            g.lchild[x], g.rchild[y] = g.rchild[y], x
            if g.lchild[x] != -1:
                g.parent[g.lchild[x]] = x
                
        chosen_node = np.random.choice(targets)
        mutate_ops = [swap]
        if newgen.lchild[chosen_node] != -1:
            mutate_ops.append(rotate_r)
        if newgen.rchild[chosen_node] != -1:
            mutate_ops.append(rotate_l)
            
        random.choice(mutate_ops)(newgen, chosen_node)
        return cls.mutate(newgen, mutate_prob=mutate_prob) \
               if random.random() < mutate_prob else newgen
    
    
    @classmethod
    def crossover(cls, gen1, gen2):
        raise NotImplementedError
    
    
    @property
    def grafting_slots(self):
        slots = []
        slots.extend([(index, 0) for index, lchild in enumerate(self.lchild) if lchild == -1])
        slots.extend([(index, 1) for index, rchild in enumerate(self.rchild) if rchild == -1])
        return slots
        
    
    @classmethod
    def graft(cls, gen1: "NodeBinaryTree", gen2: "NodeBinaryTree", slot) -> "NodeBinaryTree":
        assert not np.any(np.isin(gen2.nodes, gen1.nodes))
        nodes = np.concatenate([gen1.nodes, gen2.nodes])
        newgen = cls(nodes, initialize=False)
        offset = gen1.num_nodes
        newgen.root = gen1.root
        newgen.parent = np.concatenate([gen1.parent, gen2.parent + offset])
        newgen.lchild = np.concatenate([gen1.lchild, np.where(gen2.lchild == -1, -1, gen2.lchild + offset)])
        newgen.rchild = np.concatenate([gen1.rchild, np.where(gen2.rchild == -1, -1, gen2.rchild + offset)])
        grafting_node = slot[0]
        is_left_child = slot[1] == 0
        sub_tree_root = gen2.root + offset
        newgen.parent[sub_tree_root] = grafting_node
        if is_left_child:
            newgen.lchild[grafting_node] = sub_tree_root
        else:
            newgen.rchild[grafting_node] = sub_tree_root
        return newgen