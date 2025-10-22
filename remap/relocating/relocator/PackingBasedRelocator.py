import numpy as np
from numpy import ndarray, intp
import math
from typing import List, Union, Callable
import copy

from ..evaluation import Evaluators, FixedDisplacementOp, PackingPropertyPool, WeightedScorer
from ...DataWrapper import DataWrapper
from ...common.definition import Factory
from ...packing.solver import EASolver
from ...packing.representation import Anchor, Contour
from ...packing.representation import NodeBinaryTree as NBT
from ...packing.representation import NodeBinaryTreePackingTree as NBTPTree
from ...packing.common.definition import Verifier
from ...common.timer import Timer

timer = Timer()

class NBTPTreeVerifier(Verifier[NBTPTree]):
    def __init__(self, datawrapper: DataWrapper, anchor: Anchor):
        self._datawrapper = datawrapper
        self._anchor = anchor
    
    def __call__(self, gen):
        wrapper = self._datawrapper
        anchor = self._anchor
        blocks = gen.blocks
        haloed_w = wrapper.haloed_macro_w
        haloed_h = wrapper.haloed_macro_h
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        contour = Contour()
        contour.initialize(xl, xh, yl if anchor.bottom else yh, anchor.bottom)
        haloed_block_xl, haloed_block_yl = gen.topl(haloed_w[blocks], haloed_h[blocks], anchor.x, anchor.bottom, anchor.left, contour)
        haloed_macro_xl = wrapper.haloed_macro_x
        haloed_macro_yl = wrapper.haloed_macro_y
        haloed_macro_xl[blocks] = haloed_block_xl
        haloed_macro_yl[blocks] = haloed_block_yl
        placed_macro_mask = np.zeros(wrapper.num_macros, dtype=np.bool_)
        placed_macro_mask[:wrapper.num_macros] = ~wrapper.movable_macro_mask[:wrapper.num_macros]
        placed_macro_mask[blocks] = 1
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
        
        fixed_blocks = np.unique(gen.tree.nodes[gen.tree.fixed_mask])
        fixed_placed_macros_mask = np.isin(np.where(placed_macro_mask)[0], fixed_blocks)
        if len(fixed_blocks) > 0:
            block_xc_origin = wrapper.block_info.x.ct[fixed_blocks]
            block_yc_origin = wrapper.block_info.y.ct[fixed_blocks]
            
            displacement = np.sum(np.floor(np.abs(macro_xc[fixed_placed_macros_mask] - block_xc_origin)) +
                                  np.floor(np.abs(macro_yc[fixed_placed_macros_mask] - block_yc_origin)))
        else:
            displacement = 0.0
            
        exceeded_xl = macro_xl < xl
        exceeded_xh = macro_xh > xh
        exceeded_yl = macro_yl < yl
        exceeded_yh = macro_yh > yh
        
        violations = [
            displacement > 0,
            np.any(overlap),
            np.any(exceeded_xl),
            np.any(exceeded_xh),
            np.any(exceeded_yl),
            np.any(exceeded_yh),
        ]
        
        return not any(violations)

class PackingBasedRelocator:
    def __init__(self, datawrapper: DataWrapper, anchors: List[Anchor], eval_ops, scorer_factory):
        self.datawrapper = datawrapper
        self.anchors = anchors
        self.packings: List[Union[NBTPTree, None]] = [None] * self.num_anchors
        self.group_assignment = np.full(self.datawrapper.num_macro_groups, -1, dtype=np.int_)
        self.eval_ops = eval_ops
        self.scorer_factory = scorer_factory
        
    @property
    def num_anchors(self):
        return len(self.anchors)

    __group_areas = None
    @property
    def group_areas(self):
        if self.__group_areas is None:
            wrapper = self.datawrapper
            areas = wrapper.macro_w * wrapper.macro_h
            self.__group_areas = np.array([np.sum(areas[group]) for group in wrapper.macro_groups])
        return self.__group_areas
    
    @timer.listen("Relocating")
    def relocate(self, num_evaluation, num_evaluation_per_slot, num_pops=5):
        """ select a proper macro group and relocate it appropriately """
        unplaced_group_id = np.where(self.group_assignment == -1)[0]
        preference = self._group_selection()
        while True:
            unplaced_group_index, anchor_id = np.unravel_index(np.argmax(preference), preference.shape)
            group_id = unplaced_group_id[unplaced_group_index]
            self.group_assignment[group_id] = anchor_id
            population = self._group_mounting(group_id, anchor_id, num_pops=num_pops, num_evaluation_per_slot=num_evaluation_per_slot)
            if len(population) > 0:
                break
            self.group_assignment[group_id] = -1
            preference[unplaced_group_index, anchor_id] = -np.inf
        new_blocks = self.datawrapper.macro_groups[group_id]
        if len(new_blocks) > 1:
            final_packing = self._corner_packing_searching(population, anchor_id, num_evaluation=num_evaluation, num_pops=num_pops)
        else:
            final_packing = population[0]
        self._apply_packing(final_packing, anchor_id)

        return len(new_blocks)
    
    
    def _group_selection(self):
        wrapper = self.datawrapper
        num_macros = wrapper.num_macros
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        width, height = xh - xl, yh - yl
        half_width, half_height = width / 2, height / 2
        
        group_assignment = self.group_assignment
        
        macro_xc = wrapper.block_info.x.ct[:num_macros]
        macro_yc = wrapper.block_info.y.ct[:num_macros]
        
        unplaced_group_id = np.where(group_assignment == -1)[0]
        unplaced_groups = [wrapper.macro_groups[group_id] for group_id in unplaced_group_id]
        
        group_x = np.array([np.mean(macro_xc[group]) for group in unplaced_groups])
        group_y = np.array([np.mean(macro_yc[group]) for group in unplaced_groups])
        
        anchors = self.anchors
        anchor_x = np.array([a.x for a in anchors])
        anchor_y = np.where(np.array([a.bottom for a in anchors]), yl, yh)
        
        delta: Callable[[ndarray, ndarray], ndarray] = \
            lambda a, b: a.reshape(-1, 1) - b.reshape(1, -1)
        softmax: Callable[[ndarray], ndarray] = \
            lambda x: np.exp(x) / np.sum(np.exp(x))
            
        group_to_anchor_distance = np.sqrt(delta(group_x, anchor_x) ** 2 + delta(group_y, anchor_y) ** 2)
        preference_distance = np.vstack([softmax((np.min(dist) - dist) / math.sqrt(width ** 2 + height ** 2)) for dist in group_to_anchor_distance])

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
        ]) > io_keepout ** 2 / 4
        
        group_areas = self.group_areas
        unplaced_group_areas = group_areas[unplaced_group_id]
        preference_group_areas: ndarray = softmax(unplaced_group_areas - np.max(unplaced_group_areas))
        anchor_group_mask = np.vstack([group_assignment == index for index, _ in enumerate(anchors)])
        corner_block_area = np.array([np.sum(group_areas[mask]) if np.any(mask) else 0 for mask in anchor_group_mask])
        quarter_area = half_width * half_height
        
        if not self.datawrapper._params.io_keepout:
            corner_utilization = corner_block_area / quarter_area
        else:
            corner_utilization = (corner_block_area + corner_io_keepout_area * 8) / quarter_area

        preference_utilization: ndarray = softmax(1 - corner_utilization)
        preference = preference_distance + preference_utilization * self.datawrapper._params.preference_utilization_weight + preference_group_areas.reshape(-1, 1) * 5
        preference[:, anchor_banned] = -np.inf
        
        return preference
    
    def _group_mounting(self, group_id: intp, anchor_id: intp, *, num_pops, num_evaluation_per_slot) -> List[NBTPTree]:
        wrapper = self.datawrapper
        anchor = self.anchors[anchor_id]
        corner_packing: Union[NBTPTree, None] = self.packings[anchor_id]
        new_blocks = wrapper.macro_groups[group_id]
        
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        contour = Contour()
        contour.initialize(xl, xh, yl if anchor.bottom else yh, anchor.bottom)
        
        verifier = NBTPTreeVerifier(self.datawrapper, anchor)
        
        if len(new_blocks) == 1:
            if not corner_packing:
                pt = NBTPTree(new_blocks)
                return [pt] if verifier(pt) else []
            else:
                corner_groups = np.where(self.group_assignment == anchor_id)[0]
                corner_blocks = np.concatenate([wrapper.macro_groups[group_id] for group_id in corner_groups])
                new_blocks_tree = NBT(new_blocks)
                population = []
                for slot in corner_packing.tree.grafting_slots:
                    pt = NBTPTree(corner_blocks, initialize=False)
                    pt.tree = pt.T.graft(corner_packing.tree, new_blocks_tree, slot)
                    pt.tree.fix(corner_packing.tree.nodes)
                    population.append(pt)
                evaluators = Evaluators(
                    lambda gen: PackingPropertyPool(
                        datawrapper=self.datawrapper,
                        macros=new_blocks,
                        gen=gen,
                        anchor=anchor,
                        contour=copy.copy(contour)
                    ),
                    self.eval_ops
                )
                metrics = [evaluators(pop) for pop in population]
                scorer = self.scorer_factory()
                fitness = scorer(metrics)
                valid = np.array([verifier(pop) for pop in population])
                fitness = np.where(valid, fitness, fitness + np.max(fitness) + 1e5)
                tops = np.argsort(fitness)
                population = [population[index] for index in tops if valid[index]][:num_pops]
                return population
             
        if corner_packing:
            corner_groups = np.where(self.group_assignment == anchor_id)[0]
            corner_blocks = np.concatenate([wrapper.macro_groups[group_id] for group_id in corner_groups])
            new_blocks_tree = NBT(new_blocks)
            class GraftFactory(Factory[NBTPTree]):
                def __init__(self, slot):
                    self.slot = slot
                
                def __call__(self):
                    pt = NBTPTree(corner_blocks, initialize=False)
                    pt.tree = pt.T.graft(corner_packing.tree, new_blocks_tree, self.slot)
                    pt.tree.fix(corner_packing.tree.nodes)
                    return pt
                
            population = []
            metrics = []
            for slot in corner_packing.tree.grafting_slots:
                grafter = EASolver(factory=GraftFactory(slot), num_pops=1, num_offs=1)
                grafter.verifier = verifier
                grafter.evaluator = Evaluators(
                    lambda gen: PackingPropertyPool(
                        datawrapper=self.datawrapper,
                        macros=new_blocks,
                        gen=gen,
                        anchor=anchor,
                        contour=copy.copy(contour)
                    ),
                    self.eval_ops
                )
                grafter.scorer = self.scorer_factory()
                _population, _metrics, _, _ = grafter(num_evaluation=min(num_evaluation_per_slot, max(10, int(2000 / len(corner_packing.tree.grafting_slots)))),
                                                      verbose=False)
                population.append(_population[0])
                metrics.append(_metrics[0])
            fitness = grafter.scorer(metrics)
            valid = np.array([verifier(pop) for pop in population])
            fitness = np.where(valid, fitness, fitness + np.max(fitness) + 1e5)
            tops = np.argsort(fitness)
            population = [population[index] for index in tops if valid[index]][:num_pops]
        else:
            class PrototypeFactory(Factory[NBTPTree]):
                def __call__(self):
                    pt = NBTPTree(new_blocks)
                    return pt
            prototyper = EASolver(factory=PrototypeFactory(), num_pops=num_pops, num_offs=num_pops)
            prototyper.verifier = verifier
            prototyper.evaluator = Evaluators(
                lambda gen: PackingPropertyPool(
                    datawrapper=self.datawrapper,
                    macros=new_blocks,
                    gen=gen,
                    anchor=anchor,
                    contour=copy.copy(contour)
                ),
                eval_ops=self.eval_ops
            )
            prototyper.scorer = self.scorer_factory()
            _population, _metrics, _, _ = prototyper(num_evaluation=num_evaluation_per_slot * num_pops, verbose=False)
            population = [pop for pop in _population[:num_pops] if prototyper.verifier(pop)]
            
        return population
    
    
    def _corner_packing_searching(self, population: List[NBTPTree], anchor_id: intp, *, num_evaluation, num_pops):
        class CornerPackingFactory(Factory[NBTPTree]):
            def __init__(self, population):
                self.population = population
                self.__index = 0
                self.__mutate_flag = False
            
            def __call__(self):
                ret: NBTPTree = (NBTPTree.mutate if self.__mutate_flag else lambda _: _)(self.population[self.__index])
                # ret.tree.unfix()
                self.__index += 1
                if self.__index >= len(self.population):
                    self.__index = 0
                    self.__mutate_flag = True
                return ret
        
        wrapper = self.datawrapper
        anchor = self.anchors[anchor_id]
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        contour = Contour()
        contour.initialize(xl, xh, yl if anchor.bottom else yh, anchor.bottom)
        
        solver = EASolver(factory=CornerPackingFactory(population), num_pops=num_pops, num_offs=num_pops)
        solver.verifier = NBTPTreeVerifier(self.datawrapper, anchor)
        solver.evaluator = Evaluators(
            lambda gen: PackingPropertyPool(
                datawrapper=self.datawrapper,
                macros=population[0].blocks,
                gen=gen,
                anchor=anchor,
                contour=contour
            ),
            self.eval_ops
        )
        solver.scorer = self.scorer_factory()
        _population, _, _, curve = solver(num_evaluation=num_evaluation)
        return _population[0]
    
    
    def _apply_packing(self, packing: NBTPTree, anchor_id: np.intp):
        wrapper = self.datawrapper
        blocks = packing.blocks
        anchor = self.anchors[anchor_id]
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        contour = Contour()
        contour.initialize(xl, xh, yl if anchor.bottom else yh, anchor.bottom)

        x, y = packing.topl(
            wrapper.haloed_macro_w[blocks],
            wrapper.haloed_macro_h[blocks],
            anchor.x,
            anchor.bottom,
            anchor.left,
            contour=contour
        )

        wrapper.block_info.x.haloed_bl[blocks] = x
        wrapper.block_info.y.haloed_bl[blocks] = y
        wrapper.movable_macro_mask[blocks] = 0
        self.packings[anchor_id] = packing