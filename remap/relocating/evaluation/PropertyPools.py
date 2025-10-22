import numpy as np
from numpy import ndarray, intp, bool_, float_
from abc import abstractmethod
from typing import List
import copy

from ...DataWrapper import DataWrapper, BlockCoord
from ...common.definition import FlyweightPool, flyweight
from ...packing.representation import MPTree, Anchor, Contour
from ...packing.representation import NodeBinaryTreePackingTree as NBTPTree

class PropertyPool(FlyweightPool):
    def __init__(self, *, datawrapper: DataWrapper, macros = None, **kwargs):
        if macros is None:
            macros = np.arange(datawrapper.num_macros, dtype=intp)
        super().__init__(datawrapper=datawrapper, macros=macros, **kwargs)
    
    datawrapper: DataWrapper
    macros: ndarray
    
    @property
    @abstractmethod
    def coord_x(self) -> BlockCoord: ...
    
    @property
    @abstractmethod
    def coord_y(self) -> BlockCoord: ...
    
    @property
    @abstractmethod
    def placed_macro_mask(self) -> ndarray: ...
    
    @property
    @abstractmethod
    def corner_blocks(self) -> List[ndarray]: ...
    
    @property
    @abstractmethod
    def anchor_x(self) -> ndarray: ...
    
    @property
    @abstractmethod
    def anchor_bottom(self) -> ndarray: ...
    

class MPTreePropertyPool(PropertyPool):
    def __init__(self, *, datawrapper: DataWrapper, macros = None, gen: MPTree, **kwargs):
        super().__init__(datawrapper=datawrapper, macros=macros, gen=gen, **kwargs)
    
    gen: MPTree
    
    @property
    @flyweight
    def corner_blocks(self):
        mptree = self.gen
        corner_blocks = [pt.blocks if pt is not None else None for pt in mptree.pts]
        return corner_blocks
    
    @property
    @flyweight
    def anchor_x(self):
        mptree = self.gen
        return np.array([ac.x for ac in mptree.anchors])
    
    @property
    @flyweight
    def anchor_bottom(self):
        mptree = self.gen
        return np.array([ac.bottom for ac in mptree.anchors])
    
    @property
    @flyweight
    def coord_x(self):
        wrapper = self.datawrapper
        macro_x = self.updated_macro_x
        macro_w = wrapper.macro_w
        halo = (wrapper.block_info.halo_l, wrapper.block_info.halo_r)
        return BlockCoord(macro_x, macro_w, halo)
    
    @property
    @flyweight
    def coord_y(self):
        wrapper = self.datawrapper
        macro_y = self.updated_macro_y
        macro_h = wrapper.macro_h
        halo = (wrapper.block_info.halo_b, wrapper.block_info.halo_t)
        return BlockCoord(macro_y, macro_h, halo)
    
    @property
    @flyweight
    def updated_macro_x(self):
        return self.updated_macro_xy[0]
    
    @property
    @flyweight
    def updated_macro_y(self):
        return self.updated_macro_xy[1]
    
    @property
    @flyweight
    def updated_macro_xy(self):
        wrapper = self.datawrapper
        macro_x, macro_y = self.new_macro_xy
        updated = self.placed_macro_mask
        macro_x = np.where(updated, macro_x, wrapper.macro_x)
        macro_y = np.where(updated, macro_y, wrapper.macro_y)
        return macro_x, macro_y
    
    @property
    @flyweight
    def new_macro_xy(self):
        wrapper = self.datawrapper
        haloed_macro_x, haloed_macro_y = self.new_haloed_macro_xy
        placed_macro_mask = self.placed_macro_mask
        macro_x = np.where(placed_macro_mask, haloed_macro_x + wrapper.block_info.halo_l, haloed_macro_x)
        macro_y = np.where(placed_macro_mask, haloed_macro_y + wrapper.block_info.halo_b, haloed_macro_y)
        return macro_x, macro_y
    
    @property
    @flyweight
    def placed_macro_mask(self):
        haloed_macro_x = self.new_haloed_macro_xy[0]
        placed_macro_mask = ~np.isnan(haloed_macro_x)
        return placed_macro_mask
    
    @property
    @flyweight
    def new_haloed_macro_xy(self):
        wrapper = self.datawrapper
        gen = self.gen
        macro_w = wrapper.haloed_macro_w
        macro_h = wrapper.haloed_macro_h
        layout_bbox = wrapper.get_layout_bbox()
        macro_x, macro_y = gen.topl(macro_w, macro_h, *layout_bbox)
        return macro_x, macro_y
    
    
class PackingPropertyPool(PropertyPool):
    def __init__(self, *, datawrapper: DataWrapper, macros = None, gen: NBTPTree, anchor: Anchor, contour: Contour, **kwargs):
        super().__init__(datawrapper=datawrapper, macros=macros, gen=gen, anchor=anchor, contour=contour, **kwargs)
    
    gen: NBTPTree
    anchor: Anchor
    contour: Contour
    
    @property
    @flyweight
    def coord_x(self) -> BlockCoord:
        wrapper = self.datawrapper
        macro_x = self.updated_macro_x
        macro_w = wrapper.macro_w
        halo = (wrapper.block_info.halo_l, wrapper.block_info.halo_r)
        return BlockCoord(macro_x, macro_w, halo)
    
    @property
    @flyweight
    def coord_y(self) -> BlockCoord:
        wrapper = self.datawrapper
        macro_y = self.updated_macro_y
        macro_h = wrapper.macro_h
        halo = (wrapper.block_info.halo_b, wrapper.block_info.halo_t)
        return BlockCoord(macro_y, macro_h, halo)
    
    @property
    @flyweight
    def updated_macro_x(self):
        return self.updated_macro_xy[0]
    
    @property
    @flyweight
    def updated_macro_y(self):
        return self.updated_macro_xy[1]
    
    @property
    @flyweight
    def updated_macro_xy(self):
        wrapper = self.datawrapper
        macro_x, macro_y = self.new_macro_xy
        updated = self.placed_macro_mask
        macro_x = np.where(updated, macro_x, wrapper.macro_x)
        macro_y = np.where(updated, macro_y, wrapper.macro_y)
        return macro_x, macro_y
    
    @property
    @flyweight
    def new_macro_xy(self):
        wrapper = self.datawrapper
        haloed_macro_x, haloed_macro_y = self.new_haloed_macro_xy
        placed_macro_mask = self.placed_macro_mask
        macro_x = np.where(placed_macro_mask, haloed_macro_x + wrapper.block_info.halo_l, haloed_macro_x)
        macro_y = np.where(placed_macro_mask, haloed_macro_y + wrapper.block_info.halo_b, haloed_macro_y)
        return macro_x, macro_y
        
    @property
    @flyweight
    def placed_macro_mask(self) -> ndarray:
        wrapper = self.datawrapper
        gen = self.gen
        placed_macro_mask = np.zeros(wrapper.num_macros, dtype=bool_)
        placed_macro_mask[gen.blocks] = 1
        return placed_macro_mask
    
    @property
    @flyweight
    def corner_blocks(self) -> List[ndarray]:
        gen = self.gen
        return [gen.blocks]
    
    @property
    @flyweight
    def anchor_x(self):
        return np.array([self.anchor.x])
    
    @property
    @flyweight
    def anchor_bottom(self):
        return np.array([self.anchor.bottom])
    
    @property
    @flyweight
    def new_haloed_macro_xy(self):
        wrapper = self.datawrapper
        gen = self.gen
        anchor = self.anchor
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        contour = Contour()
        contour.initialize(xl, xh, yl if anchor.bottom else yh, anchor.bottom)
        blocks = gen.blocks
        block_w = wrapper.haloed_macro_w[blocks]
        block_h = wrapper.haloed_macro_h[blocks]
        block_x, block_y = gen.topl(
            block_w,
            block_h,
            anchor.x,
            anchor.bottom,
            anchor.left,
            contour
        )
        macro_x = np.full(wrapper.num_macros, np.nan, dtype=float_)
        macro_y = np.full(wrapper.num_macros, np.nan, dtype=float_)
        macro_x[blocks] = block_x
        macro_y[blocks] = block_y
        return macro_x, macro_y