import torch
import torch.nn as nn
import numpy as np

class CoreBackgroundPartition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_bal = config['core_bg']['lambda_balance']
        self.alpha_smooth = config['core_bg']['alpha_smooth']
        self.core_ratio = config['core_bg']['core_ratio']
        self.register_buffer('smooth_score', None)  # 存储上一时间步的平滑得分

    def forward(self, data, paths):
        num_nodes = data.x.size(0)
        if num_nodes == 0:
            return torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.float)

        # 计算加权出度
        out_deg = torch.zeros(num_nodes)
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        for i in range(edge_index.size(1)):
            u = edge_index[0, i]
            out_deg[u] += edge_weight[i]
        out_deg_norm = out_deg / (out_deg.max() + 1e-8)

        freq = torch.zeros(num_nodes)
        for p in paths:
            for v in p:
                freq[v] += 1
        freq_norm = freq / (freq.max() + 1e-8)

        phi = self.lambda_bal * out_deg_norm + (1 - self.lambda_bal) * freq_norm

        if self.smooth_score is None:
            smooth = phi
        else:
            smooth = self.alpha_smooth * self.smooth_score + (1 - self.alpha_smooth) * phi
        self.smooth_score = smooth.detach()

        sorted_vals, _ = torch.sort(smooth, descending=True)
        threshold = sorted_vals[int(num_nodes * self.core_ratio)]
        core_mask = smooth >= threshold
        bg_mask = ~core_mask

        role = core_mask.float().unsqueeze(1)  # [N, 1]

        return core_mask, bg_mask, role