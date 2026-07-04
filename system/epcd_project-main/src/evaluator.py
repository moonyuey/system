import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.device = torch.device(config['train']['device'])

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for graph_seq, path_seq, label in test_loader:
                graph_seq = [[g.to(self.device) if g is not None else None for g in seq] for seq in graph_seq]
                pred, _, _, _ = self.model(graph_seq, path_seq, None)
                all_preds.append(pred.item())
                all_trues.append(label.item())
        pred_binary = [1 if p>0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_trues, pred_binary)
        f1 = f1_score(all_trues, pred_binary)
        recall = recall_score(all_trues, pred_binary)
        return {'accuracy': acc, 'f1': f1, 'recall': recall}