import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D

import logging
import time
import math

from typing import Optional, Callable

logger = logging.getLogger('ABPlace')
        
class AngleBasedPlacer(object):
    EPSILON=1e-5
    
    def __init__(self, link_strength, macro_w, macro_h, *, layout_bbox, device="cuda:0"):
        self.xl, self.xh, self.yl, self.yh = layout_bbox
        self.width, self.height = \
            self.xh - self.xl, self.yh - self.yl
        self.half_width, self.half_height = \
            self.width / 2, self.height / 2
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self._from_numpy_to_device = lambda array: torch.from_numpy(array).to(self.device)
        
        self.lnks = self._from_numpy_to_device(link_strength)
        self.macro_w = self._from_numpy_to_device(macro_w)
        self.macro_h = self._from_numpy_to_device(macro_h)
        
        self.num_nodes = self.lnks.size(0)
        self.num_macros = self.macro_w.size(0)
        
        self.logger = logger
        
    def __delattr__(self, attr):
        if hasattr(self, attr) and getattr(self, attr) is not None:
            super().__delattr__(attr)
            
    def apply(self, node_x: ndarray, node_y: ndarray, movable_macro_mask: ndarray):
        node_x, node_y, movable_macro_mask = \
            node_x.copy(), node_y.copy(), movable_macro_mask.copy()
            
        del self.node_x
        del self.node_y
        del self.movable_macro_mask
        
        self.node_x = self._from_numpy_to_device(node_x)
        self.node_y = self._from_numpy_to_device(node_y)
        self.movable_macro_mask = self._from_numpy_to_device(movable_macro_mask).bool()
        
    def place(self, theta=None, *,
        num_iteration=1000, learning_rate=8e-2, lr_decay=0.97, l_overlap_penalty=1., 
        **kwargs
    ):
        if theta is None:
            theta = self._from_numpy_to_device(
                self.coord2theta(
                    self.node_x,
                    self.node_y,
                    self.movable_macro_mask
                )
            )
        
        theta.requires_grad_(True)
        
        self.optimizer = optim.SGD([theta], lr=learning_rate)
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=lr_decay
        )
        
        tt = time.time()

        obj_fn = lambda theta, **kwargs: self.obj_fn(
            theta, l_overlap_penalty=l_overlap_penalty, **kwargs
        )
        obj, entries = obj_fn(theta)
        # overlap_penalty = entries["overlap_penalty"].data
        # weighted_distance = entries["weighted_distance"].data
        # if overlap_penalty == 0:
        #     l_overlap_penalty = 100
        # else:
        #     l_overlap_penalty = weighted_distance / overlap_penalty / 10
        
        for it in range(num_iteration):
            converge = self.step(theta, it, l_overlap_penalty=l_overlap_penalty, **kwargs)
            # l_overlap_penalty = l_overlap_penalty * 1.0075
            if converge:
                break
        
        self.logger.info("angle-based placement takes {:.3f}s".format(time.time() - tt))
        
        return theta
    
    
    def step(self, theta, it, *, l_overlap_penalty=1., verbose=True, log_interval=100, **kwargs):
        def logfmt(it, obj, entries):
            f = lambda s, *args, **kwargs: s.format(*args, **kwargs)
            content = ", ".join([
                f("iteration {:4d}", it),
                f("Obj {:.6E}", obj.data),
                f("WeightedDistance {:.6E}", entries["weighted_distance"].data),
                f("OverlapPenalty weight {:.6E}", l_overlap_penalty),
                f("OverlapPenalty {:.6E}", entries["overlap_penalty"].data),
                f("learning rate {:.6E}", self.optimizer.param_groups[0]["lr"])
            ])
            return content
        
        obj_fn = lambda theta, **kwargs: self.obj_fn(
            theta, l_overlap_penalty=l_overlap_penalty, **kwargs
        )
        
        self.optimizer.zero_grad()
        obj, entries = obj_fn(theta)
        
        if verbose and it == 0:
            self.logger.info(logfmt(it, obj, entries))
            
        cbk_fn = kwargs.get("cbk_fn", None)
        cbk_interval = kwargs.get("cbk_interval", 100)
        
        cbfkwargs = {"theta": theta, "it": it}
        cbfkwargs.update(kwargs)
        
        if cbk_fn and it == 0:
            cbk_fn(**cbfkwargs)
            
        cbfkwargs["it"] += 1
        
        obj.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        converge = False
        
        # TODO: criteria to detect convergence and stop early
        
        reach_interval = lambda iteration, interval: \
            (iteration + 1) % interval == 0
        
        if verbose and (reach_interval(it, log_interval) or converge):
            with torch.no_grad():
                obj, entries = obj_fn(theta)
            self.logger.info(logfmt(it + 1, obj, entries))
            
        if cbk_fn and (reach_interval(it, cbk_interval) or converge):
            cbk_fn(**cbfkwargs)
            
        return converge
        
        
    def obj_fn(self, theta: Tensor, *, l_port_penalty=1., l_overlap_penalty=1., **kwargs):
        eps = type(self).EPSILON
        delta: Callable[[Tensor, Tensor], Tensor] = \
            lambda t1, t2: t1.unsqueeze(1) - t2.unsqueeze(0)
        summing: Callable[[Tensor, Tensor], Tensor] = \
            lambda t1, t2: t1.unsqueeze(1) + t2.unsqueeze(0)
            
        dmasked: Callable[[Tensor], Tensor] = \
            lambda t: t[~torch.eye(t.size(0), dtype=bool)].view(t.size(0), -1)
        softabs: Callable[[Tensor], Tensor] = \
            lambda t: torch.sqrt(t ** 2 + eps ** 2) - eps
            
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        macro_x = self.half_width  * (cos_t + 1)
        macro_y = self.half_height * (sin_t + 1)
        
        im_lnk = dmasked(self.lnks[self.movable_macro_mask][:, self.movable_macro_mask])
        im_dtx = dmasked(delta(macro_x, macro_x))
        im_dty = dmasked(delta(macro_y, macro_y))
        
        mn_lnk = self.lnks[self.movable_macro_mask][:, ~self.movable_macro_mask]
        mn_dtx = delta(macro_x, self.node_x[~self.movable_macro_mask])
        mn_dty = delta(macro_y, self.node_y[~self.movable_macro_mask])
        
        im_dst = torch.sqrt(im_dtx ** 2 + im_dty ** 2 + eps ** 2) - eps
        im_wds = torch.multiply(im_dst, im_lnk)
        
        mn_dst = torch.sqrt(mn_dtx ** 2 + mn_dty ** 2 + eps ** 2) - eps
        mn_wds = torch.multiply(mn_dst, mn_lnk)
        
        num_placed_macros = torch.sum(torch.where(self.movable_macro_mask[:self.num_macros], 0, 1))
        weighted_distance = (
            torch.sum(im_wds) +
            torch.sum(mn_wds[:, :num_placed_macros]) * 1 +
            torch.sum(mn_wds[:, num_placed_macros:]) * 1
        ) / 2
        
        macro_w = self.macro_w[self.movable_macro_mask[:self.num_macros]]
        macro_h = self.macro_h[self.movable_macro_mask[:self.num_macros]]
        
        pairs_w = summing(macro_w, macro_w) / 2
        pairs_h = summing(macro_h, macro_h) / 2
        
        H_overlap = torch.clamp_min(dmasked(pairs_w) - softabs(im_dtx), 0)
        V_overlap = torch.clamp_min(dmasked(pairs_h) - softabs(im_dty), 0)
        
        overlap_penalty = torch.sum(H_overlap * V_overlap) / 2
        
        entries = {
            "weighted_distance": (weighted_distance, 1.0              ),
            "overlap_penalty"  : (overlap_penalty  , l_overlap_penalty),
        }
        
        obj = torch.tensor(0.0).to(self.device)
        for value, weight in entries.values():
            obj += value * weight
        
        return obj, {entry: value for entry, (value, _) in entries.items()}
    
    
    def coord2theta(self, node_x: Tensor, node_y: Tensor, movable_macro_mask: Optional[Tensor] = None):
        eps = type(self).EPSILON
        if movable_macro_mask is None:
            movable_macro_mask = np.zeros(self.num_nodes, dtype=np.bool_)
            movable_macro_mask[:self.num_macros] = 1
        else:
            movable_macro_mask = movable_macro_mask.detach().cpu().bool().numpy()
            
        movable_macro_x = node_x.detach().cpu().numpy()[movable_macro_mask]
        movable_macro_y = node_y.detach().cpu().numpy()[movable_macro_mask]
        
        offset_x = movable_macro_x - self.half_width
        offset_y = movable_macro_y - self.half_height
        
        positive_mask = offset_x > 0
        floor = np.where(positive_mask, eps, -self.half_width)
        ceil  = np.where(positive_mask, self.half_width, -eps)
        tan_t = offset_y / np.clip(offset_x, floor, ceil)
        theta = np.arctan(tan_t) + np.where(positive_mask, 0, np.pi)
        
        return theta
    
    
    def theta2coord(self, theta: Tensor, *, ratio=1.):
        half_width  = self.half_width
        half_height = self.half_height
        t = theta.detach().cpu().numpy()
        real_x = half_width  * (ratio * np.cos(t) + 1)
        real_y = half_height * (ratio * np.sin(t) + 1)
        
        return real_x, real_y
    
    
    def plot_distribution(self, theta: Tensor, *, figname="distribution"):
        t = theta.detach().cpu().numpy()
        x = self.half_width  * (np.cos(t) + 1)
        y = self.half_height * (np.sin(t) + 1)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ellipse = Ellipse((self.half_width, self.half_height), self.width, self.height, color='b', fill=False)
        ax.add_artist(ellipse)
        
        macro_w = self.macro_w.detach().cpu().numpy()
        macro_h = self.macro_h.detach().cpu().numpy()
        movable_macro_mask = self.movable_macro_mask.detach().cpu().numpy()
        areas = (macro_w * macro_h)[movable_macro_mask[:self.num_macros]]
        
        k = 1
        ax.scatter(x[::k], y[::k], np.sqrt(areas / np.pi)[::k], color='b', alpha=0.5)
        
        for index, (_x, _y) in enumerate(zip(x[::k], y[::k])):
            ax.annotate(str(index + 1), xy=(_x, _y), xytext=(_x, _y))
        
        ax.set_xlim(self.width  * -0.1, self.width  * 1.1)
        ax.set_ylim(self.height * -0.1, self.height * 1.1)
        ax.set_aspect('equal', 'box')
        
        fig.savefig("{}.png".format(figname))
        plt.close(fig)
        
        
    def plot_layout(self, *, data=None, dtype="xy", figname="layout", **kwargs):
        assert data is not None
        if dtype == "xy":
            x, y = data
        elif dtype == "theta":
            x, y = self.theta2coord(data)
            
        self._plot_layout(x, y, figname=figname, **kwargs)
        
    
    def _plot_layout(self, x, y, *, print_lnk=False, figname="layout"):
        fig, ax = plt.subplots(figsize=(6, 6))
        rect = Rectangle((self.xl, self.yl), self.width, self.height, color='b', fill=False)
        ax.add_artist(rect)
        
        if isinstance(x, Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, Tensor):
            y = y.detach().cpu().numpy()
            
        radius = np.ones(self.num_nodes) * np.sqrt(self.width * self.height / 10000 / np.pi) * 2
        x_axis = radius.copy()
        y_axis = radius.copy()
        
        x_axis[:self.num_macros] = self.macro_w.detach().cpu().numpy()[:self.num_macros]
        y_axis[:self.num_macros] = self.macro_h.detach().cpu().numpy()[:self.num_macros]
        
        for index, (_x, _y, _xa, _ya) in enumerate(zip(x, y, x_axis, y_axis)):
            if index < self.num_macros:
                shape = Rectangle
                color = 'r'
            else:
                shape = Ellipse
                color = 'b'
            ax.add_artist(shape((_x - _xa / 2, _y - _ya / 2), _xa, _ya, color=color, alpha=0.5))
            
        if print_lnk:
            lnks = self.lnks.detach().cpu().numpy()
            maxlnk = np.max(lnks)
            for rx, ry, lnk_array in zip(x, y, lnks):
                for _rx, _ry, lnk in zip(x, y, lnk_array):
                    ax.add_line(Line2D([rx, _rx], [ry, _ry], linewidth=lnk / maxlnk, color='black'))
        
        ax.set_xlim(self.xl - self.width  * 0.1, self.xh + self.width  * 0.1)
        ax.set_xlim(self.yl - self.height * 0.1, self.yh + self.height * 0.1)
        ax.set_aspect('equal', 'box')
        
        fig.savefig("{}.png".format(figname))
        plt.close(fig)
        
            