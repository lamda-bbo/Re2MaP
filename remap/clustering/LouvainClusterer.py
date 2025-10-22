import networkit as nk
import numpy as np
from numpy import ndarray

from .definition import Clusterer
from dreamplace.PlaceDB import PlaceDB

class LouvainClusterer(Clusterer):
    def __init__(self, placedb: PlaceDB):
        self.placedb = placedb
        self.net_names = self.placedb.net_names
        self.net_weights = self.placedb.net_weights
        self.net2pin_map = self.placedb.net2pin_map
        self.pin2node_map = self.placedb.pin2node_map
        self.node_names = self.placedb.node_names
        self._macro_id = np.where(self.placedb.movable_macro_mask)[0]
        
        self._graph = None
        self._communities = None
        
    def __build_graph(self):
        g = nk.Graph(n=len(self.node_names), weighted=True, directed=False)

        for net_index, pins in enumerate(self.net2pin_map):
            weight = self.net_weights[net_index] if net_index < len(self.net_weights) else 1
            connected_nodes = set()
            for pin_id in pins:
                node_index = self.pin2node_map[pin_id]
                connected_nodes.add(node_index)

            if len(connected_nodes) > 1:
                virtual_node_index = g.addNode()
                for node_index in connected_nodes:
                    g.addEdge(virtual_node_index, node_index, weight)
            else:
                connected_nodes = list(connected_nodes)
                if len(connected_nodes) == 2:
                    g.addEdge(connected_nodes[0], connected_nodes[1], weight)
        return g
    
    def __generate_communities(self):
        graph = self._graph
        plm = nk.community.PLM(graph, gamma=10, par='none')
        plm.run()
        return plm.getPartition()
        
    def __load_communities(self):
        raise NotImplementedError
    
    def __communities_to_clusters(self):
        communities = self._communities
        graph = self._graph
        num_nodes = graph.numberOfNodes()
        node2community = np.array([communities.subsetOf(i) for i in range(num_nodes)], dtype=np.int_)
        
        num_macros = len(self._macro_id)
        node2cluster = np.zeros(num_nodes, dtype=np.int_)
        node2cluster[self._macro_id] = np.arange(num_macros)
        node2community[self._macro_id] = -1
        
        communities = np.unique(node2community)
        community_id = num_macros
        for community in communities:
            if community == -1:
                continue
            nodes = np.where(node2community == community)[0]
            node2cluster[nodes] = community_id
            community_id += 1
        
        clusters = []
        for index, cluster in enumerate(np.unique(node2cluster)):
            nodes = np.where(node2cluster == cluster)[0]
            clusters.append(nodes)
            assert index >= num_macros or len(nodes) == 1
        
        self._node2cluster = node2cluster
        return clusters
    
    
    def __extract_df_matrix(self):
        graph = self._graph
        node2cluster = self._node2cluster
        num_clusters = len(self._clusters)
        
        adjmat = np.zeros((num_clusters, num_clusters), dtype=np.float_)
        for u, v in graph.iterEdges():
            uc = node2cluster[u]
            vc = node2cluster[v]
            adjmat[uc, vc] += 1
            adjmat[vc, uc] += 1
        
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        return normalize(adjmat)
        
    
    def __call__(self):
        self._graph = self.__build_graph()
        self._communities = self.__generate_communities()
        self._clusters = self.__communities_to_clusters()
        self._df_matrix = self.__extract_df_matrix()
        self._available = True
        
    @property
    def clusters(self):
        return self._clusters
    
    @property
    def df_matrix(self):
        return self._df_matrix

    def save(self, file):
        raise NotImplementedError
    
    def load(self, file):
        raise NotImplementedError