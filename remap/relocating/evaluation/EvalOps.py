import numpy as np
from numpy import ndarray
from typing import Callable

import logging
logger = logging.getLogger("EvalOps")

from .PropertyPools import PropertyPool, MPTreePropertyPool, PackingPropertyPool
from ...packing.common.definition import EvalOp
from ...common.timer import Timer

timer = Timer()

class DisplacementOp(EvalOp):
    """ calculate displacement """
    name = "displacement"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        old_x, old_y = wrapper.block_info.x.ct[pool.macros], wrapper.block_info.y.ct[pool.macros]
        new_x, new_y = pool.coord_x.ct[pool.macros], pool.coord_y.ct[pool.macros]
        return np.sum(np.round(np.abs(new_x - old_x)) + np.round(np.abs(new_y - old_y)))
    
class FixedDisplacementOp(EvalOp):
    """ calculate displacement of fixed blocks in NBTPTree representation """
    name = "fixed_displacement"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PackingPropertyPool):
        wrapper = pool.datawrapper
        gen = pool.gen
        fixed_blocks = gen.tree.nodes[gen.tree.fixed_mask]
        old_x, old_y = wrapper.block_info.x.ct[fixed_blocks], wrapper.block_info.y.ct[fixed_blocks]
        new_x, new_y = pool.coord_x.ct[fixed_blocks], pool.coord_y.ct[fixed_blocks]
        return np.sum(np.round(np.abs(new_x - old_x)) + np.round(np.abs(new_y - old_y)))
    
class DataflowOp(EvalOp):
    """ calculate dataflow """
    name = "dataflow"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        num_clusters = wrapper.num_clusters
        num_macros = wrapper.num_macros
        df_matrix = wrapper.df_matrix
        placed_macro_mask = np.zeros(num_clusters, dtype=np.bool_)
        placed_macro_mask[:num_macros] = pool.placed_macro_mask
        macro_mask = np.zeros(num_clusters, dtype=np.bool_)
        macro_mask[pool.macros] = placed_macro_mask[pool.macros]
        macro_xc = pool.coord_x.ct[macro_mask[:num_macros]]
        macro_yc = pool.coord_y.ct[macro_mask[:num_macros]]
        placed_macro_xc = pool.coord_x.ct[placed_macro_mask[:num_macros]]
        placed_macro_yc = pool.coord_y.ct[placed_macro_mask[:num_macros]]
        delta: Callable[[ndarray, ndarray], ndarray] = \
            lambda x, y: x.reshape(-1, 1) - y.reshape(1, -1)
        intermacro_delta_xc = delta(macro_xc, placed_macro_xc)
        intermacro_delta_yc = delta(macro_yc, placed_macro_yc)
        
        intermacro_distance = np.sqrt(intermacro_delta_xc ** 2 + intermacro_delta_yc ** 2)
        np.fill_diagonal(intermacro_distance, 0)
        intermacro_dataflow = np.sum(np.multiply(
            intermacro_distance,
            df_matrix[macro_mask][:, placed_macro_mask]
        )) / 2
        
        cell_cluster_xc = wrapper.block_info.x.ct[num_macros:]
        cell_cluster_yc = wrapper.block_info.y.ct[num_macros:]
        macro_cell_delta_xc = delta(macro_xc, cell_cluster_xc)
        macro_cell_delta_yc = delta(macro_yc, cell_cluster_yc)
        macro_cell_distance = np.sqrt(macro_cell_delta_xc ** 2 + macro_cell_delta_yc ** 2)
        macro_cell_dataflow = np.sum(np.multiply(
            macro_cell_distance,
            df_matrix[macro_mask][:, num_macros:]
        ))
        
        return intermacro_dataflow + macro_cell_dataflow
    
    
class PeripheryCostOp(EvalOp):
    """ calculate periphery cost """
    name = "periphery_cost"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        placed_macro_mask = pool.placed_macro_mask
        macro_mask = np.zeros_like(placed_macro_mask)
        macro_mask[pool.macros] = placed_macro_mask[pool.macros]
        xc = pool.coord_x.ct[macro_mask]
        yc = pool.coord_y.ct[macro_mask]
        x_bl = pool.coord_x.bl[macro_mask]
        y_bl = pool.coord_y.bl[macro_mask]
        x_tr = pool.coord_x.tr[macro_mask]
        y_tr = pool.coord_y.tr[macro_mask]
        
        x_distances = np.where(xc < (xl + xh) / 2, x_bl - xl, xh - x_tr)
        y_distances = np.where(yc < (yl + yh) / 2, y_bl - yl, yh - y_tr)
        
        macro_losses = np.minimum(x_distances, y_distances)
        
        # log = logger.debug
        # log("=" * 80)
        # log(f"Layout boundaries: X[{xl:.2f}, {xh:.2f}], Y[{yl:.2f}, {yh:.2f}]")
        # log("-" * 37 + "Macros" + "-" * 37)
        
        # movable_indices = np.where(macro_mask)[0]
        # for i, macro_idx in enumerate(movable_indices):
        #     log(f"Macro {macro_idx:3d}: "
        #         f"Center({xc[i]:8.2f}, {yc[i]:8.2f}) | "
        #         f"BL({x_bl[i]:8.2f}, {y_bl[i]:8.2f}) | "
        #         f"TR({x_tr[i]:8.2f}, {y_tr[i]:8.2f}) | "
        #         f"XDist: {x_distances[i]:8.2f} | "
        #         f"YDist: {y_distances[i]:8.2f} | "
        #         f"Loss: {macro_losses[i]:8.2f}")
        
        # log("-" * 38 + "Loss" + "-" * 38)
        # log(f"total: {np.sum(macro_losses):.2f} | "
        #     f"average: {np.mean(macro_losses):.2f} | "
        #     f"max: {np.max(macro_losses):.2f} | "
        #     f"min: {np.min(macro_losses):.2f}")
        # log("=" * 80)
        
        return np.sum(macro_losses)
    
    
class CornerBoundingBoxOp(EvalOp):
    """ calculate corner bounding box """
    name = "corner_bounding_box"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        xl, xh, yl, yh = wrapper.get_layout_bbox()
        anchor_x = pool.anchor_x
        anchor_bottom = pool.anchor_bottom
        corners = [corner for corner in pool.corner_blocks if corner is not None and np.any(np.isin(pool.macros, corner))]
        macro_x, macro_y = pool.coord_x.bl[:wrapper.num_macros], pool.coord_y.bl[:wrapper.num_macros]
        macro_w, macro_h = wrapper.macro_w, wrapper.macro_h
        corner_x = [macro_x[corner] for corner in corners] + [anchor_x]
        corner_y = [macro_y[corner] for corner in corners] + [yl if anchor_bottom else yh]
        corner_w = [macro_w[corner] for corner in corners] + [0]
        corner_h = [macro_h[corner] for corner in corners] + [0]
        corner_xl = np.array([np.min(x) for x in corner_x])
        corner_xh = np.array([np.max(x + w) for x, w in zip(corner_x, corner_w)])
        corner_yl = np.array([np.min(y) for y in corner_y])
        corner_yh = np.array([np.max(y + h) for y, h in zip(corner_y, corner_h)])
        return np.sum(np.multiply(corner_xh - corner_xl, corner_yh - corner_yl))
    
    
class ModuleBoundingBoxOp(EvalOp):
    """ calculate module bounding box """
    name = "module_bounding_box"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        macro_x, macro_y = pool.coord_x.haloed_bl[:wrapper.num_macros], pool.coord_y.haloed_bl[:wrapper.num_macros]
        macro_w, macro_h = wrapper.haloed_macro_w, wrapper.haloed_macro_h
        total_module_bounding_box = 0
        for group in wrapper.macro_groups:
            if not np.any(np.isin(pool.macros, group)):
                continue
            group_xl = np.min([macro_x[macro_id] for macro_id in group])
            group_yl = np.min([macro_y[macro_id] for macro_id in group])
            group_xh = np.max([macro_x[macro_id] + macro_w[macro_id] for macro_id in group])
            group_yh = np.max([macro_y[macro_id] + macro_h[macro_id] for macro_id in group])
            total_module_bounding_box += (group_xh - group_xl) * (group_yh - group_yl)
        return total_module_bounding_box


class DeadspaceOp(EvalOp):
    """ calculate deadspace """
    name = "deadspace"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        placed_macro_mask = pool.placed_macro_mask
        macro_mask = np.zeros_like(placed_macro_mask, dtype=np.bool_)
        macro_mask[pool.macros] = placed_macro_mask[pool.macros]
        macro_xl, macro_yl = pool.coord_x.haloed_bl[macro_mask], pool.coord_y.haloed_bl[macro_mask]
        macro_xh, macro_yh = pool.coord_x.haloed_tr[macro_mask], pool.coord_y.haloed_tr[macro_mask]
        cols, rows, dead = wrapper.get_deadspace(macro_xl, macro_yl, macro_xh, macro_yh)
        grid_w: ndarray = cols[1:] - cols[:-1]
        grid_h: ndarray = rows[1:] - rows[:-1]
        grid_area = grid_w.reshape(-1, 1) * grid_h.reshape(1, -1)
        deadspace = np.sum(grid_area[dead >= 3])
        return deadspace


class IOPinKeepoutViolationOp(EvalOp):
    """ calculate overlap with io pin keepout region """
    name="i_o_pin_keepout_violation"
    @staticmethod
    @timer.listen(name)
    def evaluate(pool: PropertyPool):
        wrapper = pool.datawrapper
        placed_macro_mask = pool.placed_macro_mask
        macro_mask = np.zeros_like(placed_macro_mask, dtype=np.bool_)
        macro_mask[pool.macros] = placed_macro_mask[pool.macros]
        x_bl = pool.coord_x.bl[macro_mask]
        y_bl = pool.coord_y.bl[macro_mask]
        x_tr = pool.coord_x.tr[macro_mask]
        y_tr = pool.coord_y.tr[macro_mask]
        overlap = 0.0
        for xl, xh, yl, yh in zip(x_bl, x_tr, y_bl, y_tr):
            overlap += wrapper.terminal_info.calculate_overlap_area(xl, xh, yl, yh)
        return overlap