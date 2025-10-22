import numpy as np
import math
import os
from typing import List

from dreamplace.NonLinearPlace import NonLinearPlace

from .common import get_global
from .common.timer import Timer
from .DataWrapper import DataWrapper
from .ABPlace import AngleBasedPlacer
from .packing.representation import Anchor
from .relocating import (
    PackingBasedRelocator,
    EvalOp,
    DisplacementOp, PeripheryCostOp, ModuleBoundingBoxOp, CornerBoundingBoxOp, DataflowOp, IOPinKeepoutViolationOp, DeadspaceOp,
    WeightedScorer,
)

timer = Timer()

i_o_timer = timer.listen_handler("I/O")

class ReMaP:
    wrapper: DataWrapper
    
    def __init__(self, *, device="cuda:0", save_dir=None):
        self.device = device
        self.save_dir = save_dir if save_dir is not None else os.getcwd()
        self.wrapper = None
    
    def bind(self, wrapper: DataWrapper):
        self.wrapper = wrapper
        
    @property
    def num_macros(self):
        return self.wrapper.num_macros
    
    @property
    def num_clusters(self):
        return self.wrapper.num_clusters
    
    @property
    def movable_macro_mask(self):
        return self.wrapper.movable_macro_mask
    
    @timer.listen("Global Placement")
    def _global_placement(self, target_density=None):
        placedb = self.wrapper._placedb
        params = self.wrapper._params
        params.target_density = params.target_density if target_density is None else target_density
        placer = NonLinearPlace(params, placedb, timer=None)
        metrics = placer(params, placedb)
        return metrics
        
    def place(self, *, radius_ratio_init=0.9, radius_ratio_lb=0.3, target_density_init=0.92, target_density_lb=0.5, k=None):
        if k is None:
            k = math.ceil(self.num_macros / 10)
        else:
            k = max(1, k)
            
        target_density_decay = (target_density_lb / target_density_init) ** (1 / self.num_macros)
        radius_ratio_decay = (radius_ratio_lb / radius_ratio_init) ** (1 / self.num_macros)
        
        it = 0

        df_matrix = self.wrapper.df_matrix
        xl, xh, yl, yh = self.wrapper.get_layout_bbox()
        
        abpl = AngleBasedPlacer(
            df_matrix,
            self.wrapper.haloed_macro_w, self.wrapper.haloed_macro_h,
            layout_bbox=(xl, xh, yl, yh),
            device=self.device
        )
        
        anchors = [
            Anchor(x=xl, bottom=True, left=True),
            Anchor(x=xh, bottom=True, left=False),
            Anchor(x=xl, bottom=False, left=True),
            Anchor(x=xh, bottom=False, left=False),
        ]
        eval_ops: List[EvalOp] = [
            DisplacementOp(),
            PeripheryCostOp(),
            CornerBoundingBoxOp(), 
            DataflowOp(), 
            IOPinKeepoutViolationOp(),
            ModuleBoundingBoxOp(), 
            DeadspaceOp(),
        ]
        eval_ops = [
            eval_op for eval_op in eval_ops if self.wrapper._params.weights[eval_op.name] > 1e-5]
        scorer_factory = lambda: WeightedScorer(**self.wrapper._params.weights)
        mrel = PackingBasedRelocator(self.wrapper, anchors, eval_ops=eval_ops, scorer_factory=scorer_factory)
        
        ## Prototyping
        metrics = self._global_placement(target_density=target_density_init)
        
        packings = None
        
        abpl_timer = timer.listen_handler("ABPlace")
        
        while np.any(self.movable_macro_mask[:self.num_macros]):
            num_placed_macros = np.sum(~self.movable_macro_mask[:self.num_macros])
            radius_ratio = radius_ratio_init * (radius_ratio_decay ** num_placed_macros)
            self.wrapper.from_placedb()
            self.wrapper.plot("{}/iter{:02d}_0_before_abpl".format(get_global('fig_savedir'), it), packings=packings, plot_dataflow=True, plot_deadspace=True, plot_io_pin_keepout=True, plot_macro_group=True)
            self.wrapper.plot("{}/plot".format(get_global('fig_savedir')), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_macro_group=True)
            
            abpl_hdlr = next(abpl_timer)
            
            ## TODO: without ellipse: comment out below codes
            ## Angle-Based Placement
            abpl.apply(
                self.wrapper.block_info.x.ct[:],
                self.wrapper.block_info.y.ct[:],
                movable_macro_mask=self.movable_macro_mask[:self.num_clusters]
            )
            
            def abapply(theta):
                movable_macro_x, movable_macro_y = \
                    abpl.theta2coord(theta, ratio=radius_ratio)
                self.wrapper.block_info.x.haloed_ct[self.movable_macro_mask] = movable_macro_x
                self.wrapper.block_info.y.haloed_ct[self.movable_macro_mask] = movable_macro_y
                
            def abplot(*, theta, **kwargs):
                abapply(theta)
                self.wrapper.plot("{}/plot".format(get_global('fig_savedir')), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_macro_group=True)
                
            ## TODO: without abplace: num_iteration = 0
            if not self.wrapper._params.wo_ellipse:
                if self.wrapper._params.wo_abplace:
                    _num_iteration = 0
                else:
                    _num_iteration = 500
                theta = abpl.place(
                    num_iteration=_num_iteration,
                    learning_rate=1e-6,
                    lr_decay=0.98,
                    l_overlap_penalty=self.wrapper._params.l_overlap_penalty,
                    cbk_fn=abplot,
                    cbk_interval=100
                )
                abapply(theta)
            
            abpl_hdlr()
            
            self.wrapper.plot("{}/iter{:02d}_1_after_abpl".format(get_global('fig_savedir'), it), packings=packings, plot_dataflow=True, plot_deadspace=True, plot_io_pin_keepout=True, plot_macro_group=True)
            self.wrapper.plot("{}/plot".format(get_global('fig_savedir')), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_macro_group=True)
            
            ## Macro Relocating
            # mrel.relocate(k, num_evaluation=1000)
            _k = 0
            while _k < k and np.any(self.movable_macro_mask[:self.num_macros]):
                num_placed_macro = mrel.relocate(num_evaluation=100, num_evaluation_per_slot=20)
                packings = mrel.packings
                self.wrapper.plot("{}/plot".format(get_global('fig_savedir')), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_macro_group=True)
                _k += num_placed_macro
            
            self.wrapper.plot("{}/iter{:02d}_2_after_mrel".format(get_global('fig_savedir'), it), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_dataflow=True, plot_macro_group=True)
            self.wrapper.plot("{}/plot".format(get_global('fig_savedir')), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_macro_group=True)
            
            self.wrapper.update_placedb()
            
            ## Post-GP, Recursively Prototyping
            num_placed_macros = np.sum(~self.movable_macro_mask[:self.num_macros])
            target_density = target_density_init * (target_density_decay ** num_placed_macros)
            metrics = self._global_placement(target_density=target_density)
            
            it += 1
        
        self.wrapper.plot("{}/iter{:02d}_final".format(get_global('fig_savedir'), it), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_dataflow=True, plot_macro_group=True)
        self.wrapper.plot("{}/plot".format(get_global('fig_savedir')), packings=packings, plot_io_pin_keepout=True, plot_deadspace=True, plot_macro_group=True)
        i_o_hdlr = next(i_o_timer)
        self.wrapper.dump("{}/output.def".format(get_global('def_savedir')))
        self.wrapper.dump("{}/final.def".format(os.path.dirname(get_global('def_savedir'))))
        i_o_hdlr()
        return self.wrapper.macro_x, self.wrapper.macro_y