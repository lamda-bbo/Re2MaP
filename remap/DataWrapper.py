import numpy as np
from numpy import ndarray, float_
import math
from numbers import Real
from typing import Iterable, Union, List
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from .common import get_global
from .common.utils import prepare_directory, skip
from .clustering.definition import Clusterer
from .packing.representation import NBTPackingTree as NBTPTree

from .common.timer import Timer

timer = Timer()

i_o_timer = timer.listen_handler("I/O")

__placedb_backend = get_global('placedb_backend')
if __placedb_backend == "DREAMPlace.PlaceDB":
    from dreamplace.PlaceDB import PlaceDB
    from dreamplace.Params import Params


class CoordProxy(ABC):
    def __init__(self, coord: ndarray):
        self._coord = coord
        
    @property
    def data(self):
        return self._coord
    
    @property
    def coord(self):
        return self[:]
    
    @property
    def shape(self):
        return self._coord.shape
    
    def __len__(self):
        return len(self._coord)
    
    def __repr__(self):
        return f"{type(self).__name__}(coord={self._coord})"
        
    def numpy(self, dtype=None):
        return self.__array__(dtype)
        
    @abstractmethod
    def __getitem__(self, key): ...
    
    @abstractmethod
    def __setitem__(self, key, value): ...
    
    @abstractmethod
    def __array__(self, dtype=None): ...


class BottomLeftCoordProxy(CoordProxy):
    def __getitem__(self, key):
        return self._coord[key]
    
    def __setitem__(self, key, value):
        avalue = np.asarray(value, dtype=float_)
        self._coord[key] = avalue
        
    def __array__(self, dtype=None):
        return self._coord.astype(dtype)


class OffsetCoordProxy(CoordProxy):
    def __init__(self, coord: ndarray, offset: ndarray):
        super().__init__(coord)
        self._offset = offset
    
    def __repr__(self):
        return f"{type(self).__name__}(coord={self._coord}, offset={self._offset})"
    
    def __getitem__(self, key):
        return self._coord[key] + self._offset[key]
    
    def __setitem__(self, key, value):
        avalue = np.asarray(value, dtype=float_)
        self._coord[key] = avalue - self._offset[key]
        
    def __array__(self, dtype=None):
        return (self._coord + self._offset).astype(dtype)  


class CenterCoordProxy(OffsetCoordProxy):
    def __init__(self, coord: ndarray, size: ndarray):
        super().__init__(coord, size / 2.0)
        self._size = size
        
    def __repr__(self):
        return f"{type(self).__name__}(coord={self._coord}, size={self._size})"


class BlockCoord:
    def __init__(self, coord: ndarray, size: ndarray, halo: tuple):
        self._coord = coord
        self._size = size
        self._halo = halo
        self._bl = BottomLeftCoordProxy(coord)
        self._ct = CenterCoordProxy(coord, size)
        self._tr = OffsetCoordProxy(coord, size)
        self._haloed_bl = OffsetCoordProxy(coord, np.full_like(coord, -halo[0]))
        self._haloed_ct = OffsetCoordProxy(coord, halo[0] + size / 2.0)
        self._haloed_tr = OffsetCoordProxy(coord, size + halo[1])
    
    @property
    def bl(self):
        return self._bl
    
    @bl.setter
    def bl(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._bl[:] = value
    
    @property
    def ct(self):
        return self._ct
    
    @ct.setter
    def ct(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._ct[:] = value
        
    @property
    def tr(self):
        return self._tr
    
    @tr.setter
    def tr(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._tr[:] = value

    @property
    def haloed_bl(self):
        return self._haloed_bl
        
    @haloed_bl.setter
    def haloed_bl(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._haloed_bl[:] = value

    @property
    def haloed_ct(self):
        return self._haloed_ct
        
    @haloed_ct.setter
    def haloed_ct(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._haloed_ct[:] = value

    @property
    def haloed_tr(self):
        return self._haloed_tr
        
    @haloed_tr.setter
    def haloed_tr(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._haloed_tr[:] = value

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._size[:] = value

    @property
    def halo(self):
        return self._halo


class BlockInfo:
    _x: BlockCoord
    _y: BlockCoord
    
    _w: ndarray
    _h: ndarray
    
    _halo_l: Real
    _halo_r: Real
    _halo_b: Real
    _halo_t: Real
    
    def __init__(self, x_bl: ndarray, y_bl: ndarray, w: ndarray, h: ndarray, *, halo: Union[Real, Iterable[Real]] = 0):
        self.num_blocks = len(x_bl)
        if isinstance(halo, tuple):
            assert len(halo) in (2, 4)
            if len(halo) == 2:
                self._halo_l, self._halo_b = halo
                self._halo_r, self._halo_t = halo
            else:
                self._halo_l, self.halo_r, self._halo_b, self._halo_t = halo
        else:
            self._halo_l = halo
            self._halo_r = halo
            self._halo_b = halo
            self._halo_t = halo
            
        self._w = w
        self._h = h
        self._x = BlockCoord(x_bl, self._w, (self._halo_l, self._halo_r))
        self._y = BlockCoord(y_bl, self._h, (self._halo_b, self._halo_t))
    
    @property
    def halo_l(self):
        return self._halo_l
    
    @property
    def halo_r(self):
        return self._halo_r
    
    @property
    def halo_b(self):
        return self._halo_b
    
    @property
    def halo_t(self):
        return self._halo_t
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def w(self):
        return self._w
    
    @property
    def h(self):
        return self._h
    
    @property
    def haloed_w(self):
        """ haloed width """
        return self._w + self.halo_l + self.halo_r
    
    @property
    def haloed_h(self):
        """ haloed height """
        return self._h + self.halo_b + self.halo_t
    
    def __iter__(self):
        return zip(self._x._coord, self._y._coord, self.w, self.h)

class TerminalInfo:
    def __init__(self, x, y, keepout, span, *, layout_bbox):
        self._keepout = keepout
        self._span = span
        distances = np.stack([x - layout_bbox[0], layout_bbox[1] - x, y - layout_bbox[2], layout_bbox[3] - y])
        nearest = np.argmin(distances, axis=0)
        left = nearest == 0
        right = nearest == 1
        floor = nearest == 2
        ceil = nearest == 3
        w = np.zeros_like(x)
        fc_mask = np.logical_or(floor, ceil)
        lr_mask = np.logical_or(left, right)
        w[fc_mask] = span
        w[lr_mask] = keepout
        h = np.zeros_like(y)
        h[fc_mask] = keepout
        h[lr_mask] = span
        self._xl = OffsetCoordProxy(x, np.where(~left, -w, 0))
        self._xh = OffsetCoordProxy(x, np.where(~right, w, 0))
        self._yl = OffsetCoordProxy(y, np.where(~floor, -h, 0))
        self._yh = OffsetCoordProxy(y, np.where(~ceil, h, 0))
        
    @property
    def keepout(self):
        return self._keepout
        
    @property
    def xl(self):
        return self._xl
    
    @property
    def xh(self):
        return self._xh
    
    @property
    def yl(self):
        return self._yl
    
    @property
    def yh(self):
        return self._yh
    
    def calculate_overlap_area(self, xl, xh, yl, yh):
        ovlp_xl = np.clip(self.xl, xl, np.inf)
        ovlp_yl = np.clip(self.yl, yl, np.inf)
        ovlp_xh = np.clip(self.xh, -np.inf, xh)
        ovlp_yh = np.clip(self.yh, -np.inf, yh)
        ovlp_h = np.clip(ovlp_xh - ovlp_xl, 0, np.inf)
        ovlp_v = np.clip(ovlp_yh - ovlp_yl, 0, np.inf)
        return np.sum(ovlp_h * ovlp_v)

class LayoutInfo:
    _xl: Real
    _xh: Real
    _yl: Real
    _yh: Real
    _block_info: BlockInfo
    _terminal_info: TerminalInfo
    
    @property
    def xl(self): return self._xl
    @property
    def xh(self): return self._xh
    @property
    def yl(self): return self._yl
    @property
    def yh(self): return self._yh
    
    @property
    def width(self): return self.xh - self.xl
    @property
    def height(self): return self.yh - self.yl
    
    @property
    def block_info(self): return self._block_info
    
    @property
    def terminal_info(self): return self._terminal_info


class DataWrapper:
    _layout_info: LayoutInfo
    _clusterer: Clusterer
    
    def __init__(self, placedb: PlaceDB, params: Params, clusterer: Clusterer):
        self._placedb = placedb
        self._params = params
        self._clusterer = clusterer
        self._layout_info = None
        
    @timer.listen("I/O")
    def from_placedb(self):
        placedb = self._placedb
        params = self._params
        if self._layout_info is None:
            self._layout_info = LayoutInfo()
            self._layout_info._xl = placedb.xl
            self._layout_info._xh = placedb.xh
            self._layout_info._yl = placedb.yl
            self._layout_info._yh = placedb.yh
            self._macro_id = \
                np.where(placedb.movable_macro_mask)[0]
            self._num_macros = len(self._macro_id)
        
            self.clusters = self.clusterer.clusters
            self._num_clusters = len(self.clusters)
            
            self._movable_macro_mask = \
                np.zeros(self._num_clusters, dtype=np.bool_)
            self._movable_macro_mask[:self._num_macros] = 1
            
            cluster2node_names = []
            for nodes in self.clusters:
                cluster2node_names.append(
                    [placedb.node_names[node].decode(encoding='utf-8')
                     for node in nodes if node < len(placedb.node_x)])
            self._cluster2node_names = cluster2node_names
            cluster_x, cluster_y = [], []
            cluster_w, cluster_h = [], []
            for cluster in self.clusters:
                valid_nodes = cluster[cluster < len(placedb.node_x)]
                node_x = placedb.node_x[valid_nodes]
                node_y = placedb.node_y[valid_nodes]
                node_size_x = placedb.node_size_x[valid_nodes]
                node_size_y = placedb.node_size_y[valid_nodes]
                cluster_x.append(np.mean(node_x))
                cluster_y.append(np.mean(node_y))
                cluster_w.append(np.mean(node_size_x))
                cluster_h.append(np.mean(node_size_y))
            cluster_x = np.array(cluster_x, dtype=np.float_)
            cluster_y = np.array(cluster_y, dtype=np.float_)
            cluster_w = np.array(cluster_w, dtype=np.float_)
            cluster_h = np.array(cluster_h, dtype=np.float_)
            self._layout_info._block_info = BlockInfo(
                cluster_x, cluster_y, cluster_w, cluster_h, halo=(params.macro_halo_x, params.macro_halo_y))
            
            io_pin_x = placedb.node_x[placedb.num_physical_nodes - placedb.num_terminal_NIs:placedb.num_physical_nodes]
            io_pin_y = placedb.node_y[placedb.num_physical_nodes - placedb.num_terminal_NIs:placedb.num_physical_nodes]
            self._layout_info._terminal_info = TerminalInfo(
                io_pin_x, io_pin_y, params.io_pin_keepout, params.io_pin_span, layout_bbox=self.get_layout_bbox())
            
            # Deal with macro group info
            self._macro_groups = self.clusterer.macro_groups
            new_macro_groups = []

            if self._params.wo_macro_group:
                for group in self._macro_groups:
                    for macro in group:
                        new_macro_groups.append([macro])
            
            else:
                for group in self._macro_groups:
                    # 计算面积
                    group_areas = [self.block_info.w[macro] * self.block_info.h[macro] for macro in group]
                    group_area_sum = sum(group_areas)
                    layout_area = self.layout_info.width * self.layout_info.height

                    # 判断是否需要split
                    need_split = False
                    if len(group) >= 12:
                        need_split = True
                    if group_area_sum > 0.2 * layout_area:
                        need_split = True

                    if need_split and len(group) > 1:
                        half = len(group) // 2
                        new_macro_groups.append(group[:half])
                        new_macro_groups.append(group[half:])
                    else:
                        new_macro_groups.append(group)
                        
            self._macro_groups = new_macro_groups
            self._num_macro_groups = len(self._macro_groups)

            self._macro_id2group_map = {}
            for macro_group_idx, macro_group in enumerate(self._macro_groups):
                for macro in macro_group:
                    self._macro_id2group_map[macro] = macro_group_idx
            self._num_macro_groups = len(self._macro_groups)

        else:
            cluster_x, cluster_y = [], []
            clusters = []
            macro_id = []
            for cluster_id, node_names in enumerate(self._cluster2node_names):
                nodes = []
                for node_name in node_names:
                    possible_keys = [
                        node_name,
                        f"{node_name}.DREAMPlace.Shape0"
                    ]
                    for possible_key in possible_keys:
                        if possible_key in placedb.node_name2id_map:
                            node_id = placedb.node_name2id_map[possible_key]
                            break
                    else:
                        assert 0, node_name
                    if 0 <= node_id < len(placedb.node_x):
                        nodes.append(node_id)
                        if cluster_id < self.num_macros:
                            macro_id.append(node_id)
                clusters.append(np.array(nodes))
            self.clusters = clusters
            self._macro_id = np.array(macro_id)
            for cluster in clusters:
                valid_nodes = cluster[cluster < len(placedb.node_x)]
                # print(valid_nodes)
                node_x = placedb.node_x[valid_nodes]
                node_y = placedb.node_y[valid_nodes]
                cluster_x.append(np.mean(node_x))
                cluster_y.append(np.mean(node_y))
            cluster_x = np.array(cluster_x, dtype=np.float_)
            cluster_y = np.array(cluster_y, dtype=np.float_)
            self._layout_info._block_info.x.bl[:] = cluster_x
            self._layout_info._block_info.y.bl[:] = cluster_y
            
    def update_placedb(self):
        placedb = self._placedb
        params = self._params
        placedb.node_x[self._macro_id] = \
            self._layout_info._block_info.x.bl[:self.num_macros]
        placedb.node_y[self._macro_id] = \
            self._layout_info._block_info.y.bl[:self.num_macros]
        placedb.apply_force(params, placedb.node_x, placedb.node_y)
        movable_macros = self._macro_id < placedb.movable_macro_mask.shape[0]
        movable_macro_id = self._macro_id[movable_macros]
        placedb.movable_macro_mask[movable_macro_id] = \
            ~self._movable_macro_mask[:self._num_macros][movable_macros]

        i_o_hdlr = next(i_o_timer)
        output_file = "{}/output.def".format(get_global('def_savedir'))
        self.dump(output_file)
        params.def_input = output_file
        self._placedb = PlaceDB()
        self._placedb(params)
        i_o_hdlr()
    
    def _get_placed_mask(self, grid_num_x=224, grid_num_y=224):
        grid_size_x = (self.layout_info.width) / grid_num_x
        grid_size_y = (self.layout_info.height) / grid_num_y
        placed_mask = np.zeros((grid_num_x, grid_num_y), dtype=np.bool_)
        placed_macro_index = \
            np.where(~self.movable_macro_mask[:self.num_macros])[0]
        for index in placed_macro_index:
            x_bl = self.block_info.x.haloed_bl[index]
            y_bl = self.block_info.y.haloed_bl[index]
            x_tr = self.block_info.x.haloed_tr[index]
            y_tr = self.block_info.y.haloed_tr[index]
            xl = math.floor(x_bl / grid_size_x)
            yl = math.floor(y_bl / grid_size_y)
            xh = math.ceil(x_tr / grid_size_x)
            yh = math.ceil(y_tr / grid_size_y)
            placed_mask[max(xl, 0):min(xh + 1, grid_num_x),
                        max(yl, 0):min(yh + 1, grid_num_y)] = 1
        return placed_mask
    
    def _generate_blockages(self, params, *, grid_num_x=224, grid_num_y=224):
        grid_size_x = self.layout_info.width / grid_num_x
        grid_size_y = self.layout_info.height / grid_num_y
        placed_mask = self._get_placed_mask(grid_num_x, grid_num_y)
        placed_grids = np.where(placed_mask)
        blockages = []
        for gx, gy in zip(*placed_grids):
            xl = (gx * grid_size_x + self.layout_info.xl) / params.scale_factor + params.shift_factor[0]
            if gx + 1 < grid_num_x:
                xh = ((gx + 1) * grid_size_x + self.layout_info.xl) / params.scale_factor + params.shift_factor[0]
            else:
                xh = self.layout_info.xh / params.scale_factor + params.shift_factor[0]
            yl = (gy * grid_size_y + self.layout_info.yl) / params.scale_factor + params.shift_factor[1]
            if gy + 1 < grid_num_y:
                yh = ((gy + 1) * grid_size_y + self.layout_info.yl) / params.scale_factor + params.shift_factor[1]
            else:
                yh = self.layout_info.yh / params.scale_factor + params.shift_factor[1]
            blockages.append((xl, yl, xh, yh))
        return blockages

    _num_macros: int
    _macro_id: ndarray
    _movable_macro_mask: ndarray
    
    @property
    def num_macros(self):
        return self._num_macros
    
    @property
    def macro_id(self):
        return self._macro_id
    
    @property
    def num_macro_groups(self):
        return self._num_macro_groups

    @property
    def movable_macro_mask(self):
        return self._movable_macro_mask
    
    @property
    def num_clusters(self):
        return self._num_clusters
    
    @property
    def layout_info(self):
        return self._layout_info
    
    def get_layout_bbox(self):
        li = self.layout_info
        return (li.xl, li.xh, li.yl, li.yh)
    
    @property
    def block_info(self):
        return self._layout_info.block_info
    
    @property
    def terminal_info(self):
        return self._layout_info.terminal_info
    
    @property
    def cluster_x(self):
        return self.block_info.x.bl._coord
    
    @property
    def cluster_y(self):
        return self.block_info.y.bl._coord
    
    @property
    def macro_x(self):
        return self.cluster_x[:self.num_macros]
    
    @property
    def haloed_macro_x(self):
        return self.block_info.x.haloed_bl[:self.num_macros]
    
    @property
    def macro_y(self):
        return self.cluster_y[:self.num_macros]
        
    @property
    def haloed_macro_y(self):
        return self.block_info.y.haloed_bl[:self.num_macros]
    
    @property
    def macro_w(self):
        return self.block_info.w[:self.num_macros]
    
    @property
    def haloed_macro_w(self):
        return self.block_info.haloed_w[:self.num_macros]
    
    @property
    def macro_h(self):
        return self.block_info.h[:self.num_macros]
    
    @property
    def haloed_macro_h(self):
        return self.block_info.haloed_h[:self.num_macros]
    
    @property
    def clusterer(self):
        if not self._clusterer.available:
            self._clusterer()
        return self._clusterer
    
    @clusterer.setter
    def clusterer(self, value: Clusterer):
        if isinstance(value, Clusterer):
            self._clusterer = value
            
    @property
    def df_matrix(self):
        return self.clusterer.df_matrix

    @property
    def macro_groups(self):
        return self._macro_groups
    
    @property
    def macro_id2group_map(self):
        return self._macro_id2group_map
    
    
    def get_deadspace(self, macro_xl, macro_yl, macro_xh, macro_yh):
        xl, xh, yl, yh = self.get_layout_bbox()
        num_macros = len(macro_xl)
        cols, grid_xl_xh = np.unique(np.round(np.concatenate([macro_xl, macro_xh, [xl, xh]]), 2), return_inverse=True)
        grid_xl, grid_xh = grid_xl_xh[:num_macros], grid_xl_xh[num_macros:-2]
        rows, grid_yl_yh = np.unique(np.round(np.concatenate([macro_yl, macro_yh, [yl, yh]]), 2), return_inverse=True)
        grid_yl, grid_yh = grid_yl_yh[:num_macros], grid_yl_yh[num_macros:-2]
        num_cols, num_rows = len(cols), len(rows)
        dead = np.zeros((num_cols - 1, num_rows - 1), dtype=np.int_)
        dead[0, :] = 1
        dead[:, 0] = 1
        dead[-1, :] = 1
        dead[:, -1] = 1
        for gxl, gxh, gyl, gyh in zip(grid_xl, grid_xh, grid_yl, grid_yh):
            if gxl >= 1:
                dead[gxl - 1, gyl:gyh] += 1
            if gxh < num_cols - 1:
                dead[gxh, gyl:gyh] += 1
            if gyl >= 1:
                dead[gxl:gxh, gyl - 1] += 1
            if gyh < num_rows - 1:
                dead[gxl:gxh, gyh] += 1
        for gxl, gxh, gyl, gyh in zip(grid_xl, grid_xh, grid_yl, grid_yh):
            dead[gxl:gxh, gyl:gyh] = 0
        return cols, rows, dead
    
    @skip
    def plot(self, figname="plot", *, packings: List["NBTPTree"] = None, plot_skeleton=True, plot_io_pin_keepout=False, plot_deadspace=False, plot_macro_group=False, plot_dataflow=False):
        figsize = 64
        ratio = self.layout_info.width / self.layout_info.height
        figw = math.ceil(math.sqrt(figsize * ratio))
        figh = math.ceil(math.sqrt(figsize * (1 / ratio)))
        fig, ax = plt.subplots(figsize=(figw, figh))
        xl, xh, yl, yh = self.get_layout_bbox()
        width, height = self.layout_info.width, self.layout_info.height
        ax.add_artist(Rectangle((xl, yl), width, height, fill=False, color='black'))
        
        if plot_io_pin_keepout:
            io_pin_xl = self.terminal_info.xl[:]
            io_pin_xh = self.terminal_info.xh[:]
            io_pin_yl = self.terminal_info.yl[:]
            io_pin_yh = self.terminal_info.yh[:]

            for x, y, w, h in zip(io_pin_xl, io_pin_yl, io_pin_xh - io_pin_xl, io_pin_yh - io_pin_yl):
                ax.add_artist(Rectangle((x, y), w, h, color='gold', alpha=0.1))
        
        if plot_deadspace:
            placed_macro_mask = ~self.movable_macro_mask[:self.num_macros]
            macro_xl = self.block_info.x.haloed_bl[:self.num_macros][placed_macro_mask]
            macro_yl = self.block_info.y.haloed_bl[:self.num_macros][placed_macro_mask]
            macro_xh = self.block_info.x.haloed_tr[:self.num_macros][placed_macro_mask]
            macro_yh = self.block_info.y.haloed_tr[:self.num_macros][placed_macro_mask]
            cols, rows, dead = self.get_deadspace(macro_xl, macro_yl, macro_xh, macro_yh)
            for col, (colx, colw) in enumerate(zip(cols[:-1], cols[1:] - cols[:-1])):
                for row, (rowy, rowh) in enumerate(zip(rows[:-1], rows[1:] - rows[:-1])):
                    if dead[col, row] >= 3:
                        ax.add_artist(Rectangle((colx, rowy), colw, rowh, facecolor='none', edgecolor='red', hatch='/', fill=False))
                    else:
                        ax.add_artist(Rectangle((colx, rowy), colw, rowh, facecolor='none', edgecolor='red', linestyle='--', fill=False, alpha=0.3))
                        
        if plot_macro_group:
            colors = [
                '#e41a1c',
                '#377eb8',
                '#4daf4a',
                '#984ea3',
                '#ff7f00',
                '#ffff33',
                '#a65628',
                '#f781bf',
                '#999999',
                '#1b9e77',
            ]
            for macro_group_index, macro_group in enumerate(self.macro_groups):
                macro_x = self.macro_x[macro_group]
                macro_y = self.macro_y[macro_group]
                macro_w = self.macro_w[macro_group]
                macro_h = self.macro_h[macro_group]
                for x, y, w, h, movable in zip(macro_x, macro_y, macro_w, macro_h, self.movable_macro_mask[macro_group]):
                    if not movable:
                        ax.add_artist(Rectangle((x, y), w, h, color=colors[macro_group_index % 10], alpha=0.5))
                    else:
                        ax.add_artist(Rectangle((x, y), w, h, color='b', alpha=0.5))
                    
                    text_x = x + w / 2
                    text_y = y + h / 2
                    ax.text(text_x, text_y, str(macro_group_index), 
                            ha='center', va='center', fontsize=8, 
                            color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                if np.all(~self.movable_macro_mask[macro_group]):
                    group_xl = np.min(macro_x)
                    group_yl = np.min(macro_y)
                    group_xh = np.max(macro_x + macro_w)
                    group_yh = np.max(macro_y + macro_h)
                    ax.add_artist(Rectangle((group_xl, group_yl), group_xh - group_xl, group_yh - group_yl, fill=False, color=colors[macro_group_index % 10], linewidth=3, alpha=0.8))
        else:
            for x, y, w, h, movable in zip(self.macro_x, self.macro_y, self.macro_w, self.macro_h, self.movable_macro_mask[:self.num_macros]):
                if not movable:
                    ax.add_artist(Rectangle((x, y), w, h, color='r', alpha=0.5))
                else:
                    ax.add_artist(Rectangle((x, y), w, h, color='b', alpha=0.5))
            
        if plot_dataflow:
            threshold = np.percentile(self.df_matrix, 80) # show connection with top 20% dataflow strength
            min_width = 0.1
            max_width = 2.0
            max_df = np.max(self.df_matrix) + 1e-5
            macro_xc = self.block_info.x.ct[:self.num_macros]
            macro_yc = self.block_info.y.ct[:self.num_macros]
            cluster_xc = self.block_info.x.ct[:self.num_clusters]
            cluster_yc = self.block_info.y.ct[:self.num_clusters]
            for macro_id, (x, y) in enumerate(zip(macro_xc, macro_yc)):
                for cluster_id, (_x, _y, df) in enumerate(zip(cluster_xc, cluster_yc, self.df_matrix[macro_id])):
                    if macro_id != cluster_id and df > threshold:
                        linewidth = min_width + (max_width - min_width) * (df / max_df)
                        ax.add_line(Line2D([x, _x], [y, _y], linewidth, color='black', alpha=0.1))
        
        if packings is not None and plot_skeleton:
            arrowprops = dict(
                arrowstyle='-|>',
                linewidth=3,
                mutation_scale=9,
                alpha=0.6
            )
            lcprops = dict(
                color='black',
                **arrowprops,
            )
            rcprops = dict(
                color='red',
                **arrowprops,
            )
            for packing in packings:
                if packing is None:
                    continue
                tree = packing.tree
                nodes = packing.tree.nodes
                for index, (lc, rc) in enumerate(zip(tree.lchild, tree.rchild)):
                    parent = nodes[index]
                    parent_x = self.block_info.x.ct[parent]
                    parent_y = self.block_info.y.ct[parent]
                    if lc != -1:
                        lchild = nodes[lc]
                        lchild_x = self.block_info.x.ct[lchild]
                        lchild_y = self.block_info.y.ct[lchild]
                        ax.annotate('', (lchild_x, lchild_y), (parent_x, parent_y), arrowprops=lcprops)
                    if rc != -1:
                        rchild = nodes[rc]
                        rchild_x = self.block_info.x.ct[rchild]
                        rchild_y = self.block_info.y.ct[rchild]
                        ax.annotate('', (rchild_x, rchild_y), (parent_x, parent_y), arrowprops=rcprops)
        
        cell_cluster_x = self.cluster_x[self.num_macros:]
        cell_cluster_y = self.cluster_y[self.num_macros:]
        ax.scatter(cell_cluster_x, cell_cluster_y, 1, color='b', alpha=0.5)
        
        ax.set_xlim(min(xl, np.min(self.block_info.x.bl[:self.num_macros])), max(xh, np.max(self.block_info.x.tr[:self.num_macros])))
        ax.set_ylim(min(yl, np.min(self.block_info.y.bl[:self.num_macros])), max(yh, np.max(self.block_info.y.tr[:self.num_macros])))
        # ax.set_aspect('equal', 'box')
        
        file_name = f"{figname}"
        prepare_directory(file_name)
        fig.savefig(file_name)
        plt.close(fig)
    
    def dump(self, output_file):
        placedb = self._placedb
        params = self._params
        prepare_directory(output_file)
        placedb.write(params, output_file, blockages=self._generate_blockages(params))
        