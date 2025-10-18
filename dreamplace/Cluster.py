# import networkit as nk
# import pdb

# class GraphBuilder:
#     def __init__(self, placedb):
#         self.net_names = placedb.net_names
#         self.net_weights = placedb.net_weights
#         self.net2pin_map = placedb.net2pin_map

#         self.pin2node_map = placedb.pin2node_map
#         self.node_names = placedb.node_names

#         # Networkit 使用整数作为节点索引，创建一个带权图
#         self.graph = nk.Graph(n=len(self.node_names), weighted=True, directed=False)

#     def add_nodes(self):
#         # 在 networkit 中，默认已经有了足够的节点，无需添加
#         pass

#     def add_edges(self):
#         # 添加边，基于 nets 和 pins
#         node_name_to_index = {name: i for i, name in enumerate(self.node_names)}

#         for net_index, pins in enumerate(self.net2pin_map):
#             weight = self.net_weights[net_index] if net_index < len(self.net_weights) else 1  # 使用默认权重1如果没有提供
#             connected_nodes = set()
#             for pin_id in pins:
#                 node_index = self.pin2node_map[pin_id]
#                 connected_nodes.add(node_index)  # 直接使用节点索引

#             if len(connected_nodes) > 1:
#                 # 创建一个虚拟节点
#                 virtual_node_index = self.graph.addNode()
#                 # 连接虚拟节点到所有相关的节点
#                 for node_index in connected_nodes:
#                     self.graph.addEdge(virtual_node_index, node_index, weight)
#             else:
#                 # 直接连接这两个节点（如果只有两个节点的话）
#                 connected_nodes = list(connected_nodes)
#                 if len(connected_nodes) == 2:
#                     self.graph.addEdge(connected_nodes[0], connected_nodes[1], weight)

#     def build_graph(self):
#         self.add_nodes()
#         self.add_edges()
#         return self.graph

import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pdb

class Cluster():
    def __init__(self, placedb, cluster_file, macro_group_file, df_file):
        self.placedb = placedb
        self.net_names = self.placedb.net_names
        self.net_weights = self.placedb.net_weights
        self.net2pin_map = self.placedb.net2pin_map
        self.pin2node_map = self.placedb.pin2node_map
        self.node_names = self.placedb.node_names
        self.node_name2id_map = self.placedb.node_name2id_map
        self.movable_macro_mask = self.placedb.movable_macro_mask

        self.macro_group_file = macro_group_file
        self.df_file = df_file
        self.cluster_file = cluster_file

        self.cell_cluster_ids = None                                        # 所有 node 到 cluster id 的映射
        self.cluster2node_ids = defaultdict(list)                           # cluster id 到 cluster 包含所有 cell id 的映射

        self.connect_matrix = None                                          # num_cell_clusters * num_cell_clusters 的cluster连接
        self.updated_connect_matrix = None                                  # num_maro * (num_macro + num_cell_clusters) macro 和 cell cluster的连接矩阵
        self.df_matrix = None                                               # num_maro * (num_macro + num_cell_clusters) macro 和 macro 的数据流矩阵

        self.num_cell_clusters = None                                       # 所有cell cluster的数量
        self.macro_list = []                                                # 把macros按顺序放在一个list中
        self.macro_id2list_map = {}                                         # 从 macro id 到 macro list 排位的映射
        self.macro_group_ids = -np.ones(len(self.node_names), dtype=int)    # 把上述list中的macro分成group
        self.group2macro_ids = defaultdict(list)                            # 把macro按照group排列

        self.build_cluster()
        self.init_connect_matrix()
        self.build_group()
        self.init_df_matrix()

        self.visualize_matrix(self.df_matrix)

    def build_cluster(self):
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
                self.cluster2node_ids[cluster_id].append(node_id)  # 维护cluster到node_id的映射

        # 检查所有 cell 是否都分配了 cluster_id
        num_nodes = len(self.node_names)
        self.cell_cluster_ids = np.zeros(num_nodes, dtype=int)
        for i in range(num_nodes):
            if i in cell_id2cluster_id:
                self.cell_cluster_ids[i] = cell_id2cluster_id[i]
            else:
                raise RuntimeError

    def init_connect_matrix(self):
        """
        初始化 cluster 间的连接矩阵
        """
        if self.cell_cluster_ids is None:
            raise RuntimeError("请先调用 build_cluster 方法！")

        num_clusters = int(self.cell_cluster_ids.max()) + 1
        self.connect_matrix = np.zeros((num_clusters, num_clusters), dtype=int)

        for net in self.net2pin_map:
            clusters = set()
            for pin in net:
                parent_node_id = self.pin2node_map[pin]
                parent_cluster_id = self.cell_cluster_ids[parent_node_id]
                clusters.add(parent_cluster_id)

            clusters = list(clusters)
            for m in clusters:
                for n in clusters:
                    self.connect_matrix[m][n] += 1

        pdb.set_trace()
    
    def build_group(self):
        with open(self.macro_group_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['clustering_results']
        macro_group_id = 0
        for cluster_dict in data:
            if cluster_dict['type'] == 'Macro':
                for macro_name in cluster_dict['macros']:
                    macro_id = self.node_name2id_map[macro_name]
                    self.macro_group_ids[macro_id] = macro_group_id
                    self.group2macro_ids[macro_group_id].append(macro_id)
                macro_group_id += 1

    def init_df_matrix(self):

        for macro_id in np.where(self.movable_macro_mask==1)[0]:
            self.macro_list.append(macro_id)
            self.macro_id2list_map[macro_id] = len(self.macro_list) - 1

        # 初始化 df matrix 和 updated connect matrix, size 都是 num_maro * (num_macro + num_cell_clusters)
        self.num_cell_clusters = int(self.cell_cluster_ids.max()) + 1
        self.df_matrix = np.zeros((len(self.macro_list), len(self.macro_list) + self.num_cell_clusters))
        self.updated_connect_matrix = np.zeros((len(self.macro_list), len(self.macro_list) + self.num_cell_clusters))

        # 读取 data_connections.json
        with open(self.df_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['data_connections']

        # 针对 macro-cell 连接关系, 提取 df matrix
        for macro_pin_dict in data["macro_pins_and_regs"]:
            pin_name = macro_pin_dict["macro_pin"]
            macro_name = "/".join(pin_name.split("/")[:-1])
            macro_id = self.node_name2id_map[macro_name]
            parent_macro_list_id = self.macro_id2list_map[macro_id]
            for hop in macro_pin_dict["hops"]:
                num_hop = hop["hop"]
                cost = 8 * (0.5 ** num_hop)
                for cell_name in hop["registers"]:
                    cell_id = self.node_name2id_map[cell_name]
                    cell_cluster_id = self.cell_cluster_ids[cell_id]
                    self.df_matrix[parent_macro_list_id][cell_cluster_id] += cost

        # 针对 macro-macro 连接关系, 提取 df matrix
        for macro_pin_dict in data["macro_pins_and_macros"]:
            pin_name = macro_pin_dict["macro_pin"]
            macro_name = "/".join(pin_name.split("/")[:-1])
            macro_id = self.node_name2id_map[macro_name]
            parent_macro_list_id = self.macro_id2list_map[macro_id]
            for hop in macro_pin_dict["hops"]:
                num_hop = hop["hop"]
                cost = 8 * (0.5 ** num_hop)
                for macro_name_ in hop["macros"]:
                    macro_id = self.node_name2id_map[macro_name_]
                    macro_list_id = self.macro_id2list_map[macro_id]
                    self.df_matrix[parent_macro_list_id][self.num_cell_clusters + macro_list_id] += cost
                    self.df_matrix[macro_list_id][self.num_cell_clusters + parent_macro_list_id] += cost

        # 对于 cellcluster 的连接关系矩阵，我们遍历所有 macro id, 根据每个 macro 所属的 cellcluster, 
        # 找出其连接的 cluster, 并添加到最终 updated matrix 中。
        for macro_id in self.macro_list:
            parent_cluster_id = self.cell_cluster_ids[macro_id]
            macro_rank = self.macro_id2list_map[macro_id]
            # pdb.set_trace()
            self.updated_connect_matrix[macro_rank][:self.num_cell_clusters] += self.connect_matrix[parent_cluster_id]


    def visualize_matrix(self, arr):

        plt.figure(figsize=(8, 8))  # 设置画布大小
        plt.imshow(arr[:,-10:], cmap='viridis', interpolation='nearest')
        plt.colorbar()  # 显示颜色条
        plt.title("connect_matrix")
        print(np.sum(arr))
        plt.savefig("matrix.pdf")

    def save(self, file):
        raise NotImplementedError
    
    def load(self, file):
        raise NotImplementedError
    