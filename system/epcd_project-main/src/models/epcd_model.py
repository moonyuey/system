import torch
import torch.nn as nn
import torch.nn.functional as F
from .core_background import CoreBackgroundPartition
from .dual_encoder import DualPathEncoder
from .cross_attention import CrossPathAttention

class EPCDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core_bg = CoreBackgroundPartition(config)
        self.path_encoder = DualPathEncoder(config)
        self.cross_attn = CrossPathAttention(config)
        # BiGRU for temporal aggregation
        self.gru = nn.GRU(input_size=config['model']['path_repr_dim'],
                          hidden_size=config['model']['path_repr_dim'],
                          num_layers=1, bidirectional=True, batch_first=True)
        self.gru_proj = nn.Linear(2*config['model']['path_repr_dim'], config['model']['path_repr_dim'])
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['path_repr_dim'], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, graph_sequence, path_sequence, labels=None):

        T = len(graph_sequence)
        e_t_list = []
        path_loss = 0
        node_loss = 0


        self.core_bg.smooth_score = None

        for t in range(T):
            data = graph_sequence[t]
            paths = path_sequence[t]
            if data is None or len(paths) == 0:
                continue
            # 核心-背景划分
            core_mask, bg_mask, role = self.core_bg(data, paths)
            # 对每条路径编码
            path_reprs = []
            neg_ratios = []
            path_labels = []
            node_labels = []
            for p in paths:
                if len(p) < 2:
                    continue

                emo = data.x[p, -3:]  # [L,3]
                neg_ratio = (emo[:, 0] > 0.5).float().mean().item()
                neg_ratios.append(neg_ratio)
                z_p, beta = self.path_encoder(p, data, role)
                if z_p is None:
                    continue
                path_reprs.append(z_p)

                core_in_path = any(role[v].item() == 1 for v in p)
                if core_in_path and neg_ratio > 0.6:
                    path_labels.append(1.0)
                else:
                    path_labels.append(0.0)

                node_label = []
                for v in p:
                    is_core = role[v].item() == 1
                    neg_conf = data.x[v, -3]  # 负向概率
                    if is_core and neg_conf > 0.6:
                        node_label.append(1.0)
                    else:
                        node_label.append(0.0)
                node_labels.append(node_label)
            if len(path_reprs) == 0:
                continue
            e_t, alpha, lp, ln = self.cross_attn(path_reprs, neg_ratios,
                                                 path_labels if labels is not None else None,
                                                 node_labels if labels is not None else None)
            path_loss += lp
            node_loss += ln
            if e_t is not None:
                e_t_list.append(e_t)

        if len(e_t_list) == 0:

            return torch.tensor(0.5, device=next(self.parameters()).device), 0, 0

        e_seq = torch.stack(e_t_list, dim=0).unsqueeze(0)  # [1, T', dim]
        gru_out, _ = self.gru(e_seq)  # [1, T', 2*dim]
        global_repr = gru_out[:, -1, :]  # 取最后时间步
        global_repr = self.gru_proj(global_repr)  # [1, dim]

        logit = self.classifier(global_repr).squeeze()
        pred = logit  # 概率值

        class_loss = 0
        if labels is not None:
            class_loss = F.binary_cross_entropy(pred, labels.float())

        total_loss = class_loss + self.config['explain']['lambda_path'] * path_loss + self.config['explain']['lambda_node'] * node_loss
        return pred, total_loss, alpha, e_t_list