import torch
import networkx as nx
from collections import deque

class PathExtractor:
    def __init__(self, config):
        self.max_len = config['path']['max_path_length']
        self.top_k_ratio = config['path']['top_k_seed_ratio']

    def extract_paths_from_graph(self, data):
        if data is None:
            return []
        edge_index = data.edge_index.numpy()
        num_nodes = data.x.size(0)
        if num_nodes == 0 or edge_index.shape[1] == 0:
            return []
        # 构建NetworkX有向图（带权重，但路径抽取只考虑拓扑）
        G = nx.DiGraph()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0][i], edge_index[1][i]
            G.add_edge(u, v)
        # 计算加权出度（用于选择种子节点）
        out_deg = torch.zeros(num_nodes)
        for i, (u, v) in enumerate(edge_index.T.tolist()):
            out_deg[u] += data.edge_weight[i].item()
        # 选择种子节点：top-k 出度
        k = max(1, int(num_nodes * self.top_k_ratio))
        if k > num_nodes:
            k = num_nodes
        top_nodes = torch.topk(out_deg, k).indices.tolist()
        paths = []
        # 对每个种子执行DFS/BFS，提取所有长度<=max_len的简单路径（从种子出发）
        for seed in top_nodes:
            self._bfs_paths(G, seed, [], paths)
        return paths

    def _bfs_paths(self, G, source, current_path, all_paths):
        current_path = current_path + [source]
        if len(current_path) > self.max_len:
            return
        # 只要路径长度>=2就保存（至少一条边）
        if len(current_path) >= 2:
            all_paths.append(current_path.copy())
        # 获取后继节点
        if source in G:
            for neighbor in G.successors(source):
                if neighbor not in current_path:  # 避免环
                    self._bfs_paths(G, neighbor, current_path, all_paths)

    def extract_all(self, graph_sequence):
        all_paths_per_session = []
        for data in graph_sequence:
            paths = self.extract_paths_from_graph(data)
            all_paths_per_session.append(paths)
        return all_paths_per_session