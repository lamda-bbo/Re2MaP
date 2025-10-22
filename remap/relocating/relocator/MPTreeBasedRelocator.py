import numpy as np
from numpy import ndarray
from typing import Tuple, List, Type, TypeVar, Generic, Callable

from ...DataWrapper import DataWrapper
from ...common.definition import Factory
from ...packing.common.definition import Evaluator, Verifier, Scorer
from ...packing.representation import PackingTree, Anchor, MPTree, BBTSPackingTree, NBTPackingTree, MPTreeMetaClass
from ...packing.solver import EASolver

import logging

logger = logging.getLogger("MaReloc")

PT_TYPE = TypeVar('PT_TYPE', bound=PackingTree)

class PackingTreeFactory(Generic[PT_TYPE], Factory[PT_TYPE]):
    pttype: Type[PT_TYPE]
    
    def __init__(self, blocks):
        self.blocks = blocks
        
    @property
    def num_blocks(self):
        return len(self.blocks)
    
    def __call__(self) -> PT_TYPE:
        if self.num_blocks == 0:
            return None
        pt = type(self).pttype(self.blocks)
        return pt

class BBTSPackingTreeFactory(PackingTreeFactory[BBTSPackingTree]):
    pttype = BBTSPackingTree
    
class NBTPackingTreeFactory(PackingTreeFactory[NBTPackingTree]):
    pttype = NBTPackingTree
    
class MPTreeFactoryMetaClass(Generic[PT_TYPE]):
    def __init__(meta, anchors: List[Anchor], ptfmc: Type[PackingTreeFactory[PT_TYPE]]):
        meta.anchors = anchors
        meta.ptfmc = ptfmc
        meta.pttype = ptfmc.pttype
    
    def make(meta):
        class PT_MPTreeFactory(Factory[MPTree[PT_TYPE]]):
            def __init__(self, blocks_list: List):
                self.ptfs = [meta.ptfmc(blocks) for blocks in blocks_list]
            
            def __call__(self, **kwargs):
                mpt = MPTreeMetaClass[PT_TYPE].make(meta.pttype)(meta.anchors, self.ptfs, **kwargs)
                return mpt
        
        return PT_MPTreeFactory


class MPTreeVerifier(Verifier[MPTree]):
    def __init__(self, datawrapper: DataWrapper):
        self._datawrapper = datawrapper
    
    def __call__(self, gen):
        wrapper = self._datawrapper
        haloed_w = wrapper.haloed_macro_w
        haloed_h = wrapper.haloed_macro_h
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        haloed_macro_xl, haloed_macro_yl = gen.topl(haloed_w, haloed_h, xl, xh, yl, yh)
        placed_macro_mask = ~np.isnan(haloed_macro_xl)
        macro_xl = haloed_macro_xl[placed_macro_mask] + wrapper.block_info.halo_l
        macro_yl = haloed_macro_yl[placed_macro_mask] + wrapper.block_info.halo_b
        macro_w = wrapper.macro_w[placed_macro_mask]
        macro_h = wrapper.macro_h[placed_macro_mask]
        macro_xh = macro_xl + macro_w
        macro_yh = macro_yl + macro_h
        macro_xc = macro_xl + macro_w / 2
        macro_yc = macro_yl + macro_h / 2
        delta: Callable[[ndarray, ndarray], ndarray] = \
            lambda a, b: a.reshape(-1, 1) - b.reshape(1, -1)
        summing: Callable[[ndarray, ndarray], ndarray] = \
            lambda a, b: a.reshape(-1, 1) + b.reshape(1, -1)
        horizontal_overlap = np.abs(delta(macro_xc, macro_xc)) - summing(macro_w, macro_w) / 2 < 0
        vertical_overlap = np.abs(delta(macro_yc, macro_yc)) - summing(macro_h, macro_h) / 2 < 0
        overlap = np.logical_and(horizontal_overlap, vertical_overlap)
        np.fill_diagonal(overlap, 0)
        
        exceeded_xl = macro_xl < xl
        exceeded_xh = macro_xh > xh
        exceeded_yl = macro_yl < yl
        exceeded_yh = macro_yh > yh
        
        return not (
            np.any(overlap)
            or np.any(exceeded_xl)
            or np.any(exceeded_xh)
            or np.any(exceeded_yl)
            or np.any(exceeded_yh)
        )
            
            
class MPTreeBasedRelocator(Generic[PT_TYPE]):
    def __init__(self, datawrapper: DataWrapper, mptfmc: MPTreeFactoryMetaClass[PT_TYPE], *, evaluator, scorer, verifier=None, **kwargs):
        self.datawrapper = datawrapper
        self.mptfmc = mptfmc
        self.evaluator = evaluator
        self.scorer = scorer
        self.verifier = verifier
        self.macro_assignment = np.full(self.datawrapper.num_macros, -1, dtype=np.int_)
        self.placed_groups = None
        
    def relocate(self, num_to_relocate, num_evaluation):
        wrapper = self.datawrapper
        macro_x, macro_y, self.macro_assignment = \
            self._relocate(
                num_to_relocate,
                wrapper,
                self.macro_assignment,
                self.verifier,
                self.evaluator,
                self.scorer,
                num_evaluation)

        num_macros = wrapper.num_macros
        wrapper.block_info.x.haloed_bl[:num_macros] = macro_x[:num_macros]
        wrapper.block_info.y.haloed_bl[:num_macros] = macro_y[:num_macros]
        
        
    def _relocate(self,
        num2relocate,
        wrapper: DataWrapper,
        macro_assignment: ndarray,
        verifier: Verifier,
        evaluator: Evaluator,
        scorer: Scorer,
        num_evaluation=1000,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        num_macros = wrapper.num_macros
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        width, height = xh - xl, yh - yl
        half_width, half_height = width / 2, height / 2
        
        unplaced_macro_mask = wrapper.movable_macro_mask[:num_macros]
        macro_assignment = macro_assignment.copy()
        
        macro_xc = wrapper.block_info.x.ct[:num_macros]
        macro_yc = wrapper.block_info.y.ct[:num_macros]
        
        unplaced_group_id, unplaced_groups = \
            tuple(zip(*[(group_id, group) for group_id, group in enumerate(wrapper.macro_groups)
                        if np.all(group != -1) and np.all(unplaced_macro_mask[group])]))
        unplaced_group_id = np.array(unplaced_group_id)
        num_unplaced_groups = len(unplaced_groups)
        group_x = np.array([np.mean(macro_xc[group]) for group in unplaced_groups])
        group_y = np.array([np.mean(macro_yc[group]) for group in unplaced_groups])
        
        anchors = self.mptfmc.anchors
        anchor_x = np.array([a.x for a in anchors])
        anchor_y = np.where(np.array([a.bottom for a in anchors]), yl, yh)
        
        delta: Callable[[ndarray, ndarray], ndarray] = \
            lambda a, b: a.reshape(-1, 1) - b.reshape(1, -1)
        softmax: Callable[[ndarray], ndarray] = \
            lambda x: np.exp(x) / np.sum(np.exp(x))
            
        group_to_anchor_distance = np.sqrt(delta(group_x, anchor_x) ** 2 + delta(group_y, anchor_y) ** 2)
        preference_distance = np.vstack([softmax(np.min(dist) - dist) for dist in group_to_anchor_distance])

        corner_io_keepout_area = np.array([
            wrapper.terminal_info.calculate_overlap_area(
                ax - half_width,  ax + half_width,
                ay - half_height, ay + half_height
            )
            for ax, ay in zip(anchor_x, anchor_y)
        ])
        io_keepout = wrapper.terminal_info.keepout
        anchor_banned = np.array([
            wrapper.terminal_info.calculate_overlap_area(
                ax - io_keepout, ax + io_keepout,
                ay - io_keepout, ay + io_keepout
            )
            for ax, ay in zip(anchor_x, anchor_y)
        ]) > io_keepout ** 2
        group_banned = np.zeros(num_unplaced_groups, dtype=np.bool_)
        
        areas = wrapper.macro_w * wrapper.macro_h
        anchor_macro_mask = np.vstack([macro_assignment == index for index, _ in enumerate(anchors)])
        corner_block_area = np.array([np.sum(areas[mask]) if np.any(mask) else 0 for mask in anchor_macro_mask])
        quarter_area = half_width * half_height
        
        num_placed_macros = 0
        while num_placed_macros < num2relocate:
            corner_utilization = (corner_block_area + corner_io_keepout_area) / quarter_area
            preference_utilization: ndarray = softmax(1 - corner_utilization)
            preference = preference_distance * 0.9 + preference_utilization * 0.0
            preference[group_banned, :] = -np.inf
            preference[:, anchor_banned] = -np.inf
            
            group_index, anchor = np.unravel_index(np.argmax(preference), preference.shape)
            group_id = unplaced_group_id[group_index]

            relative_macros = wrapper.macro_groups[group_id]
            group_banned[group_index] = True
            unplaced_macro_mask[relative_macros] = False
            corner_block_area[anchor] += np.sum(relative_macros)
            macro_assignment[relative_macros] = anchor
            num_placed_macros += len(relative_macros)
        
        blocks_list = [np.where(macro_assignment == index)[0].tolist() for index, _ in enumerate(anchors)]
        mptf = self.mptfmc.make()(blocks_list)
        solver = EASolver(factory=mptf, num_pops=10, num_offs=5)
        solver.verifier = verifier
        solver.evaluator = evaluator
        solver.scorer = scorer
        population, metrics, fitness, trace = solver(num_evaluation=num_evaluation)
        best_pop = population[0]
        new_macro_x, new_macro_y = best_pop.topl(
            wrapper.haloed_macro_w, wrapper.haloed_macro_h, xl, xh, yl, yh)
        macro_x = np.where(unplaced_macro_mask, wrapper.haloed_macro_x, new_macro_x)
        macro_y = np.where(unplaced_macro_mask, wrapper.haloed_macro_y, new_macro_y)
        return macro_x, macro_y, macro_assignment