import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class DualPathEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gcn_hid = config['model']['gcn_hidden_dim']
        self.lstm_hid = config['model']['lstm_hidden_dim']
        self.out_dim = config['model']['path_repr_dim']

        self.gcn1 = GCNConv(config['model']['node_feat_dim'] + 1, self.gcn_hid)  # +1 for role
        self.gcn2 = GCNConv(self.gcn_hid, self.out_dim)

        self.lstm = nn.LSTM(input_size=3, hidden_size=self.lstm_hid, bidirectional=True, batch_first=True)
        self.lstm_proj = nn.Linear(2*self.lstm_hid, self.out_dim)

        # 门控融合参数
        self.W_g = nn.Linear(2*self.out_dim, 1)
        self.b_g = nn.Parameter(torch.zeros(1))

        self.q = nn.Parameter(torch.randn(self.out_dim))

    def forward(self, path_nodes, data, role_indicator):

        if len(path_nodes) < 2:
            return None, None
        # 提取路径节点特征
        node_feat = data.x  # [N, feat_dim]
        # 拼接角色
        feat_with_role = torch.cat([node_feat, role_indicator], dim=1)  # [N, feat_dim+1]
        # 路径诱导子图
        sub_nodes = path_nodes
        # 构造子图边（从原图中提取）
        edge_index = data.edge_index
        node_set = set(sub_nodes)
        # 找子图内的边
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for i, (u, v) in enumerate(edge_index.t()):
            if u.item() in node_set and v.item() in node_set:
                mask[i] = True
        sub_edge_index = edge_index[:, mask]
        # 重映射节点索引
        old2new = {old: new for new, old in enumerate(sub_nodes)}
        sub_edge_index = torch.tensor([[old2new[u.item()], old2new[v.item()]] for u, v in sub_edge_index.t()], dtype=torch.long).t()
        sub_feat = feat_with_role[sub_nodes]  # [L, feat_dim+1]
        sub_data = Data(x=sub_feat, edge_index=sub_edge_index)
        # 如果无边，则自行添加自环
        if sub_data.edge_index.size(1) == 0:
            sub_data.edge_index = torch.tensor([[i, i] for i in range(len(sub_nodes))], dtype=torch.long).t()


        x = self.gcn1(sub_data.x, sub_data.edge_index)
        x = F.relu(x)
        x = self.gcn2(x, sub_data.edge_index)
        H_s = x  # [L, out_dim]

        emotion_feat = node_feat[sub_nodes, -3:]  # [L, 3]
        lstm_out, _ = self.lstm(emotion_feat.unsqueeze(0))  # [1, L, 2*hid]
        H_e = self.lstm_proj(lstm_out.squeeze(0))  # [L, out_dim]

        concat = torch.cat([H_s, H_e], dim=1)  # [L, 2*out_dim]
        g = torch.sigmoid(self.W_g(concat) + self.b_g)  # [L, 1]
        H = g * H_s + (1 - g) * H_e  # [L, out_dim]

        # 注意力池化
        # 计算注意力权重 beta
        attn_score = torch.matmul(H, self.q)  # [L]
        beta = F.softmax(attn_score, dim=0)  # [L]
        z_p = torch.sum(beta.unsqueeze(1) * H, dim=0)  # [out_dim]

        return z_p, beta