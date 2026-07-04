import torch
import numpy as np
from sklearn.metrics import precision_score

class ExplainEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.device = torch.device(config['train']['device'])
        self.top_k = config['explain']['top_k_paths']

    def evaluate_path_consistency(self, test_loader, num_samples=100):
        """计算Precision@k"""
        self.model.eval()
        sampled = 0
        all_precisions = {k: [] for k in self.top_k}
        with torch.no_grad():
            for graph_seq, path_seq, label in test_loader:
                if label.item() != 1:
                    continue
                graph_seq = [[g.to(self.device) if g is not None else None for g in seq] for seq in graph_seq]
                pred, _, alpha, _ = self.model(graph_seq, path_seq, None)
                if pred.item() < 0.5:
                    continue

        return {k: np.mean(prec) for k, prec in all_precisions.items()}