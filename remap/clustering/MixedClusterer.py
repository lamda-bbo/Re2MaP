import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pdb

from .definition import Clusterer
from ..common import get_global
from dreamplace.PlaceDB import PlaceDB

class MixedClusterer(Clusterer):
    def __init__(self, placedb: PlaceDB, params):
        self.placedb = placedb
        self.net_names = self.placedb.net_names
        self.net_weights = self.placedb.net_weights
        self.net2pin_map = self.placedb.net2pin_map
        self.pin2node_map = self.placedb.pin2node_map
        self.node_names = self.placedb.node_names
        self.node_name2id_map = self.placedb.node_name2id_map
        self.movable_macro_mask = self.placedb.movable_macro_mask
        self.params = params

        self.design_name = get_global('design_name')
        base_dir = f"benchmarks/clustering_results/{self.design_name}"
        self.macro_group_file = (self.params.macro_group_file_path or f"{base_dir}/clustering_results.json")
        self.df_file = (self.params.df_file_path or f"{base_dir}/dataflow_results.json")
        self.cluster_file = (self.params.cluster_file_path or f"{base_dir}/cluster_map.txt")

        # 内部状态变量
        self._cell_cluster_ids = None                                        # 所有 node 到 cluster id 的映射
        self._cluster2node_ids = defaultdict(list)                           # cluster id 到 cluster 包含所有 cell id 的映射
        self._connect_matrix = None                                          # num_cell_clusters * num_cell_clusters 的cluster连接
        self._updated_connect_matrix = None                                  # num_maro * (num_macro + num_cell_clusters) macro 和 cell cluster 的连接矩阵
        self._df_matrix = None                                               # num_maro * (num_macro + num_cell_clusters) macro 和 macro / cell cluster 的数据流矩阵
        self._clusters = None                                                # 聚类结果列表
        self._num_cell_clusters = None                                       # 所有cell cluster的数量
        self._macro_list = []                                                # 把macros按顺序放在一个list中
        self._macro_id2list_map = {}                                         # 从 macro id 到 macro list 排位的映射
        self._macro_group_ids = -np.ones(len(self.node_names), dtype=int)    # 把上述list中的macro分成group
        self._group2macro_ids = defaultdict(list)                            # 把macro按照group排列
        self._final_matrix = None

        self._alpha = 1
        self._beta = 1

        self._available = False
        
    def __build_cluster(self):
        """
        从文件读取每个 cell 的聚类编号，填充 cell_id2cluster_id 和 cell_cluster_ids
        """
        with open(self.cluster_file, 'r') as f:
            cell_id2cluster_id = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2 or len(parts) > 2:
                    continue
                node_name = parts[0]
                cluster_id = int(parts[1])

                node_id = self.node_name2id_map[node_name]
                cell_id2cluster_id[node_id] = cluster_id
                self._cluster2node_ids[cluster_id].append(node_id)  # 维护cluster到node_id的映射

        # 检查所有 cell 是否都分配了 cluster_id
        num_nodes = len(self.node_names)
        self._cell_cluster_ids = np.zeros(num_nodes, dtype=int)
        for i in range(num_nodes):
            if i in cell_id2cluster_id:
                self._cell_cluster_ids[i] = cell_id2cluster_id[i]
            else:
                raise RuntimeError

    def __init_connect_matrix(self):
        """
        初始化 cluster 间的连接矩阵
        """
        if self._cell_cluster_ids is None:
            raise RuntimeError("请先调用 __build_cluster 方法！")

        num_clusters = int(self._cell_cluster_ids.max()) + 1
        self._connect_matrix = np.zeros((num_clusters, num_clusters), dtype=int)

        for net in self.net2pin_map:
            clusters = set()
            for pin in net:
                parent_node_id = self.pin2node_map[pin]
                parent_cluster_id = self._cell_cluster_ids[parent_node_id]
                clusters.add(parent_cluster_id)

            clusters = list(clusters)
            for m in clusters:
                for n in clusters:
                    self._connect_matrix[m][n] += 1
    
    def __build_group(self):
        with open(self.macro_group_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['clustering_results']
        macro_group_id = 0
        for cluster_dict in data:
            if cluster_dict['type'] == 'Macro':
                for macro_name in cluster_dict['macros']:
                    macro_id = self.node_name2id_map[macro_name]
                    self._macro_group_ids[macro_id] = macro_group_id
                    self._group2macro_ids[macro_group_id].append(macro_id)
                macro_group_id += 1

    def __init_df_matrix(self):
        for macro_id in np.where(self.movable_macro_mask==1)[0]:
            self._macro_list.append(macro_id)
            self._macro_id2list_map[macro_id] = len(self._macro_list) - 1

        # 初始化 df matrix 和 updated connect matrix, size 都是 num_maro * (num_macro + num_cell_clusters)
        self._num_cell_clusters = int(self._cell_cluster_ids.max()) + 1
        dimension = len(self._macro_list) + self._num_cell_clusters
        self._df_matrix = np.zeros((dimension, dimension))
        self._updated_connect_matrix = np.zeros((dimension, dimension))

        # 读取 data_connections.json
        with open(self.df_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['data_connections']

        # 针对 macro-cell 连接关系, 提取 df matrix
        for macro_pin_dict in data["macro_pins_and_regs"]:
            pin_name = macro_pin_dict["macro_pin"]
            macro_name = "/".join(pin_name.split("/")[:-1])
            macro_id = self.node_name2id_map[macro_name]
            parent_macro_list_id = self._macro_id2list_map[macro_id]
            for hop in macro_pin_dict["hops"]:
                num_hop = hop["hop"]
                cost = 8 * (0.5 ** num_hop)
                for cell_name in hop["registers"]:
                    cell_id = self.node_name2id_map[cell_name]
                    cell_cluster_id = self._cell_cluster_ids[cell_id]
                    self._df_matrix[parent_macro_list_id][cell_cluster_id] += cost

        # 针对 macro-macro 连接关系, 提取 df matrix
        for macro_pin_dict in data["macro_pins_and_macros"]:
            pin_name = macro_pin_dict["macro_pin"]
            macro_name = "/".join(pin_name.split("/")[:-1])
            macro_id = self.node_name2id_map[macro_name]
            parent_macro_list_id = self._macro_id2list_map[macro_id]
            for hop in macro_pin_dict["hops"]:
                num_hop = hop["hop"]
                cost = 8 * (0.5 ** num_hop)
                for macro_name_ in hop["macros"]:
                    macro_id = self.node_name2id_map[macro_name_]
                    macro_list_id = self._macro_id2list_map[macro_id]
                    self._df_matrix[parent_macro_list_id][self._num_cell_clusters + macro_list_id] += cost
                    self._df_matrix[macro_list_id][self._num_cell_clusters + parent_macro_list_id] += cost

        # 对于 cellcluster 的连接关系矩阵，我们遍历所有 macro id, 根据每个 macro 所属的 cellcluster, 
        # 找出其连接的 cluster, 并添加到最终 updated matrix 中。
        for macro_id in self._macro_list:
            parent_cluster_id = self._cell_cluster_ids[macro_id]
            macro_rank = self._macro_id2list_map[macro_id]
            self._updated_connect_matrix[macro_rank][:self._num_cell_clusters] += self._connect_matrix[parent_cluster_id]

    def __merge_clusters(self):

        self._clusters = []
        my_movable_macro_mask = np.zeros(self.placedb.num_physical_nodes, dtype=bool)
        my_movable_macro_mask[:self.placedb.num_movable_nodes] = self.placedb.movable_macro_mask

        for i in range(len(self._macro_list)):
            self._clusters.append(np.array([self._macro_list[i]], dtype=np.int_))
        for i in range(len(self._cluster2node_ids)):
            # print(len(self._cluster2node_ids[i]))
            nodes = np.array(self._cluster2node_ids[i], dtype=np.int_)
            self._clusters.append(nodes)
            # mask = my_movable_macro_mask[nodes]
            # self._clusters.append(nodes[~mask])
            
    def __merge_df_matrix(self):
        # 对connect_matrix归一化
        self._updated_connect_matrix = (self._updated_connect_matrix - self._updated_connect_matrix.min()) / (self._updated_connect_matrix.max() - self._updated_connect_matrix.min())

        # 对df_matrix归一化
        self._df_matrix = (self._df_matrix - self._df_matrix.min()) / (self._df_matrix.max() - self._df_matrix.min())

        self._final_matrix = self._updated_connect_matrix * self._alpha + self._df_matrix * self._beta

    def __call__(self):
        """一旦调用，所有属性变为可用"""
        self.__build_cluster()
        self.__init_connect_matrix()
        self.__build_group()
        self.__init_df_matrix()
        self.__merge_clusters()
        self.__merge_df_matrix()
        self._available = True
        
    @property
    def clusters(self):
        """返回聚类结果列表"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._clusters
    
    @property
    def df_matrix(self):
        """返回数据流矩阵"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._final_matrix
    
    @property
    def cell_cluster_ids(self):
        """返回cell到cluster的映射"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._cell_cluster_ids
    
    @property
    def connect_matrix(self):
        """返回cluster连接矩阵，注意这个基本没用！主要用updated_connect_matrix"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._connect_matrix
    
    @property
    def updated_connect_matrix(self):
        """返回更新后的连接矩阵"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._updated_connect_matrix
    
    @property
    def macro_list(self):
        """返回macro列表"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._macro_list
    
    @property
    def macro_id2list_map(self):
        """返回macro id到list索引的映射"""
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        return self._macro_id2list_map
    
    @property
    def macro_groups(self):
        """
        返回macro group, 是一个np array的列表
        [np.array([macro_id_1, macro_id_2, ...]),
         np.array([macro_id_3, macro_id_4, ...]),
         ...]
        """
        if not self._available:
            raise RuntimeError("请先调用 __call__ 方法！")
        macro_group_ls = []
        for i in range(len(self._group2macro_ids)):
            macro_group = self._group2macro_ids[i]
            current_group = [self._macro_id2list_map[macro_id] for macro_id in macro_group]
            macro_group_ls.append(np.array(current_group))
        return macro_group_ls


    def visualize_matrix(self, arr):
        """可视化矩阵"""
        plt.figure(figsize=(8, 8))  # 设置画布大小
        plt.imshow(arr[:,-10:], cmap='viridis', interpolation='nearest')
        plt.colorbar()  # 显示颜色条
        plt.title("connect_matrix")
        plt.savefig("matrix.pdf")

    def save(self, file):
        """保存聚类结果到文件"""
        raise NotImplementedError
    
    def load(self, file):
        """从文件加载聚类结果"""
        raise NotImplementedError
    