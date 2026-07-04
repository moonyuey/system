import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['train']['device'])
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['train']['learning_rate'])
        self.criterion = torch.nn.BCELoss()
        self.early_stop_patience = config['train']['early_stop_patience']

    def train_epoch(self, data_loader):

        self.model.train()
        total_loss = 0
        for graph_seq, path_seq, label in data_loader:

            graph_seq = [[g.to(self.device) if g is not None else None for g in seq] for seq in graph_seq]
            label = label.to(self.device).float()  # 确保为浮点数
            self.optimizer.zero_grad()
            pred, loss, _, _ = self.model(graph_seq, path_seq, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def validate(self, data_loader):

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for graph_seq, path_seq, label in data_loader:
                graph_seq = [[g.to(self.device) if g is not None else None for g in seq] for seq in graph_seq]
                pred, _, _, _ = self.model(graph_seq, path_seq, None)
                preds.append(pred.item())
                trues.append(label.item())
        # 二值化预测
        pred_binary = [1 if p > 0.5 else 0 for p in preds]
        acc = accuracy_score(trues, pred_binary)
        f1 = f1_score(trues, pred_binary)
        recall = recall_score(trues, pred_binary)
        return acc, f1, recall, preds, trues

    def train(self, train_loader, val_loader, epochs):
        best_val_f1 = 0.0
        patience = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc, val_f1, val_recall, _, _ = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f}")
            # 早停判断
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_epcd_model.pt')
                print(f"  -> Best model saved (F1 = {val_f1:.4f})")
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        print("Training finished. Best Val F1: {:.4f}".format(best_val_f1))

    def evaluate(self, test_loader):

        acc, f1, recall, preds, trues = self.validate(test_loader)
        print(f"Test Results: Acc={acc:.4f}, F1={f1:.4f}, Recall={recall:.4f}")
        return acc, f1, recall, preds, trues