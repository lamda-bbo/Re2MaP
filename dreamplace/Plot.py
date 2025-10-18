import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pdb
import re  # 导入正则表达式库
import pickle
import random
import networkit as nk

data = {
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9]
}

def build_graph_from_adjacency_matrix(adj_matrix):
    G = nk.Graph(n=len(adj_matrix), weighted=True)
    size = len(adj_matrix)
    for i in range(size):
        for j in range(i + 1, size):  # 只处理上三角部分
            if adj_matrix[i][j] != 0:
                G.addEdge(i, j, adj_matrix[i][j])
    return G

def second_order_adjacency_matrix(G, decay_rate):
    """ 生成并返回二阶邻接矩阵 """
    size = G.numberOfNodes()
    second_order_adj_matrix = np.zeros((size, size))

    for node in range(size):
        neighbors = list(G.iterNeighbors(node))
        for neighbor in neighbors:
            second_neighbors = list(G.iterNeighbors(neighbor))
            for sec_neighbor in second_neighbors:
                if sec_neighbor != node:  # 排除自身
                    first_edge_weight = G.weight(node, neighbor)
                    second_edge_weight = G.weight(neighbor, sec_neighbor)
                    weaker_strength = min(first_edge_weight, second_edge_weight) * decay_rate
                    # 更新矩阵，保存最大的二阶连接强度
                    second_order_adj_matrix[node][sec_neighbor] += weaker_strength

    return second_order_adj_matrix

def third_order_adjacency_matrix(G, decay_rate):
    """ 生成并返回三阶邻接矩阵 """
    size = G.numberOfNodes()
    third_order_adj_matrix = np.zeros((size, size))

    for node in range(size):
        first_neighbors = list(G.iterNeighbors(node))
        for first_neighbor in first_neighbors:
            second_neighbors = list(G.iterNeighbors(first_neighbor))
            for second_neighbor in second_neighbors:
                if second_neighbor != node:  # 排除从自身直接返回的路径
                    third_neighbors = list(G.iterNeighbors(second_neighbor))
                    for third_neighbor in third_neighbors:
                        if third_neighbor != node and third_neighbor != first_neighbor:  # 避免重复节点
                            first_edge_weight = G.weight(node, first_neighbor)
                            second_edge_weight = G.weight(first_neighbor, second_neighbor)
                            third_edge_weight = G.weight(second_neighbor, third_neighbor)
                            weakest_strength = min(first_edge_weight, second_edge_weight, third_edge_weight) * decay_rate

                            third_order_adj_matrix[node][third_neighbor] += weakest_strength

    return third_order_adj_matrix

def extract_adjacency_matrix(placedb, communities, g, macro_id):

    node_x = placedb.node_x
    node_y = placedb.node_y
    node_size_x = placedb.node_size_x
    node_size_y = placedb.node_size_y

    community_ids = [communities.subsetOf(i) for i in range(g.numberOfNodes())]
    clusters = []
    node2cluster = [-1] * g.numberOfNodes()

    # 先为每个macro创建一个单独的集群
    index = 0
    for i in macro_id:
        clusters.append([i])
        node2cluster[i] = index
        index += 1

    # 为非macro节点按社区ID分组
    # 使用临时字典来帮助分组
    community_to_cluster_index = {}
    for i, comm_id in enumerate(community_ids):
        if i not in macro_id:
            if comm_id not in community_to_cluster_index:
                clusters.append([])
                community_to_cluster_index[comm_id] = index
                index += 1
            cluster_index = community_to_cluster_index[comm_id]
            clusters[cluster_index].append(i)
            node2cluster[i] = cluster_index

    num_clusters = len(clusters)
    adjacency_matrix = np.zeros((num_clusters, num_clusters), dtype=int)

    for u, v in g.iterEdges():
        u_cluster = node2cluster[u]
        v_cluster = node2cluster[v]
        if u_cluster != -1 and v_cluster != -1 and u_cluster != v_cluster:
            adjacency_matrix[u_cluster][v_cluster] += 1
            adjacency_matrix[v_cluster][u_cluster] += 1

        
    # 计算每个cluster的平均坐标
    cluster_x = []
    cluster_y = []

    for cluster in clusters:
        sum_x = 0
        sum_y = 0
        valid_nodes = 0
        for node in cluster:
            if node < len(node_x):  # 确保不超出坐标数组的长度
                sum_x += node_x[node]
                sum_y += node_y[node]
                valid_nodes += 1
        if valid_nodes > 0:
            avg_x = sum_x / valid_nodes
            avg_y = sum_y / valid_nodes
        else:
            avg_x = None  # 如果没有有效节点，可能需要指定一个默认值或None
            avg_y = None
        cluster_x.append(avg_x)
        cluster_y.append(avg_y)

    # 输出矩阵和统计信息
    print("Number of all clusters:", num_clusters)
    print("Adjacency Matrix:")
    print(adjacency_matrix)
    # print("Cluster X coordinates:", cluster_x)
    # print("Cluster Y coordinates:", cluster_y)
    G = build_graph_from_adjacency_matrix(np.array(adjacency_matrix))

    # 获取二阶邻接矩阵
    second_order_matrix = second_order_adjacency_matrix(G, 0.25)

    # 为了防止调试耗时，暂时先不运行third order matrix
    third_order_matrix = None
    # third_order_matrix = third_order_adjacency_matrix(G, 0.125)

    # 打印二阶邻接矩阵
    # print("Second Order Adjacency Matrix:")
    # print(second_order_matrix)
    # print("Third Order Adjacency Matrix:")
    # print(third_order_matrix)

    # pdb.set_trace()

    # 输出每个macro连接的cell cluster id
    # macro_cluster_indices = range(len(macro_id))
    # print("Macro Connections:")
    # for idx in macro_cluster_indices:
    #     connected_clusters = [i for i in range(len(clusters)) if adjacency_matrix[idx][i] > 0]
    #     print(f"Macro Node {clusters[idx][0]} (Cluster {idx}) is connected to clusters: {connected_clusters}")

    return adjacency_matrix, clusters, node2cluster, cluster_x, cluster_y, second_order_matrix, third_order_matrix

def plot_clusters(placedb, cluster2name, macro_indices, cell_clusters, adjacency_matrix, macro_id):
    
    node_x, node_y = placedb.node_x, placedb.node_y
    plt.figure(figsize=(10, 8))

    plotted_clusters = set()  # 用来跟踪已经绘制过的clusters

    for macro_index in macro_indices:
        # 遍历每个cluster，检查它是否与当前macro有连接
        for cluster_index, connections in enumerate(adjacency_matrix[macro_index]):
            if connections > 0 and cluster_index not in plotted_clusters:
                node_indices = cell_clusters[int(cluster2name[cluster_index].split('_')[1])-len(macro_id)]
                cluster_color = (random.random(), random.random(), random.random())
                # 绘制该cluster中的所有cells
                for node_index in node_indices:
                    plt.scatter(node_x[node_index], node_y[node_index], color=cluster_color, s=20)
                plotted_clusters.add(cluster_index)  # 标记为已绘制
                print(cluster_index, cluster_color)

    # 绘制宏
    for macro_index in macro_id:
        plt.scatter(node_x[macro_index], node_y[macro_index], color='black', marker='x', s=100)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cell Clusters Connected to Macros')
    plt.xlim(0, max(node_x) + 10)  # 设置坐标轴范围以确保所有点可见
    plt.ylim(0, max(node_y) + 10)
    plt.grid(True)
    plt.savefig("clusters.png")
    plt.show()

def plot_ports(placedb, communities, commu_set):
    # 创建一个正则表达式模式匹配 "p" 后跟一个或多个数字
    pattern = re.compile(r'^p\d+$')
    node_name2id_map = placedb.node_name2id_map
    node_x = placedb.node_x
    node_y = placedb.node_y
    port_num = 0
    # 使用字典来存储每个社区的节点坐标
    community_coords = {}

    # 遍历字典，找到所有匹配的节点名称
    for node_name, node_id in node_name2id_map.items():
        if pattern.match(node_name):
            community_id = communities.subsetOf(node_id)
            print(community_id, commu_set)
            if community_id not in commu_set:
                continue
            # 确保 node_id 在数组范围内
            if node_id < len(node_x) and node_id < len(node_y):
                port_num += 1
                if community_id not in community_coords:
                    community_coords[community_id] = {'x': [], 'y': []}
                community_coords[community_id]['x'].append(node_x[node_id])
                community_coords[community_id]['y'].append(node_y[node_id])
    
    return community_coords

    # # 检查是否有符合条件的节点
    # if community_coords:
    #     # 绘制节点
    #     plt.figure(figsize=(10, 6))
    #     for community_id, coords in community_coords.items():
    #         plt.scatter(coords['x'], coords['y'], label=f'Community {community_id}')
    #     max_x = max(max(coords['x']) for coords in community_coords.values())
    #     max_y = max(max(coords['y']) for coords in community_coords.values())
    #     plt.xlim(-500, max_x+500)
    #     plt.ylim(-500, max_y+500)
    #     plt.title('Node Positions by Community')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     # plt.legend()
    #     plt.grid(True)
    #     plt.savefig("ports.pdf")
    #     plt.show()
    # else:
    #     print("No matching nodes found.")

    # print("port_num: ", port_num)




