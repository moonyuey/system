import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossPathAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['model']['path_repr_dim']
        self.W_s = nn.Linear(self.dim, self.dim)
        self.v = nn.Parameter(torch.randn(self.dim))
        self.b_s = nn.Parameter(torch.zeros(self.dim))

    def forward(self, path_reprs, path_neg_ratios, path_labels=None, node_labels=None):

        M = len(path_reprs)
        if M == 0:
            return None, None, 0, 0
        path_reprs = torch.stack(path_reprs, dim=0)  # [M, dim]
        neg_ratios = torch.tensor(path_neg_ratios, device=path_reprs.device)

        # 路径得分
        s = self.v * torch.tanh(self.W_s(path_reprs) + self.b_s)  # [M, dim] -> [M]
        s = s.sum(dim=1)

        # 情绪增强因子
        gamma = 1 + neg_ratios / (neg_ratios.max() + 1e-8)
        alpha = F.softmax(s * gamma, dim=0)  # [M]

        # 加权聚合
        e_t = torch.sum(alpha.unsqueeze(1) * path_reprs, dim=0)  # [dim]

        loss_path = 0
        loss_node = 0
        if path_labels is not None:

            loss_path = F.binary_cross_entropy(alpha, path_labels.float(), reduction='sum') / M
        if node_labels is not None and hasattr(self, 'node_betas'):
            pass
        return e_t, alpha, loss_path, loss_node