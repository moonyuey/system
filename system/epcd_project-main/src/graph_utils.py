import torch
import numpy as np
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

def normalize_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
    deg = degree(edge_index[0], num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    row, col = edge_index
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight

def get_induced_subgraph(data, node_mask):

    if isinstance(node_mask, list):
        node_mask = torch.tensor(node_mask, dtype=torch.bool)
    nodes = torch.where(node_mask)[0].tolist()
    if len(nodes) == 0:
        return None
    # 提取边
    edge_index = data.edge_index
    # 保留两个端点都在节点集中的边
    mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub_edge_index = edge_index[:, mask]
    # 重映射节点索引
    node_map = {old: new for new, old in enumerate(nodes)}
    if sub_edge_index.size(1) > 0:
        sub_edge_index = torch.tensor([[node_map[u.item()], node_map[v.item()]]
                                       for u, v in sub_edge_index.T], dtype=torch.long).T
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)
    sub_x = data.x[node_mask]
    sub_edge_weight = data.edge_weight[mask] if hasattr(data, 'edge_weight') else None
    sub_data = Data(x=sub_x, edge_index=sub_edge_index)
    if sub_edge_weight is not None:
        sub_data.edge_weight = sub_edge_weight
    return sub_data

def compute_degree_stats(data):

    num_nodes = data.x.size(0)
    out_deg = torch.zeros(num_nodes)
    in_deg = torch.zeros(num_nodes)
    if data.edge_index.size(1) > 0:
        for i in range(data.edge_index.size(1)):
            u = data.edge_index[0, i]
            v = data.edge_index[1, i]
            w = data.edge_weight[i] if hasattr(data, 'edge_weight') else 1.0
            out_deg[u] += w
            in_deg[v] += w
    norm_out = out_deg / (out_deg.sum() + 1e-8)
    return torch.stack([out_deg, in_deg, norm_out], dim=1)