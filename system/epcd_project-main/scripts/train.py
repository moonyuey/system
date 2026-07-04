import sys
sys.path.append('..')
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from src.data_loader import DataLoader as Preprocessor
from src.path_extractor import PathExtractor
from src.models.epcd_model import EPCDModel
from src.trainer import Trainer
from src.evaluator import Evaluator

class SessionDataset(Dataset):
    def __init__(self, graph_sequences, path_sequences, labels):
        self.graph_sequences = graph_sequences
        self.path_sequences = path_sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.graph_sequences[idx], self.path_sequences[idx], self.labels[idx]

def main():
    with open('../configs/epcd_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 预处理数据（若已存在则加载）
    preprocessor = Preprocessor(config)
    try:
        graph_sequences, labels = preprocessor.load_processed()
    except:
        graph_sequences, labels = preprocessor.preprocess_all()

    # 提取路径
    path_extractor = PathExtractor(config)
    path_sequences = path_extractor.extract_all(graph_sequences)

    # 划分数据集
    from sklearn.model_selection import train_test_split
    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)  # 0.8*0.2=0.16

    train_graphs = [graph_sequences[i] for i in train_idx]
    train_paths = [path_sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_graphs = [graph_sequences[i] for i in val_idx]
    val_paths = [path_sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_graphs = [graph_sequences[i] for i in test_idx]
    test_paths = [path_sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_dataset = SessionDataset(train_graphs, train_paths, train_labels)
    val_dataset = SessionDataset(val_graphs, val_paths, val_labels)
    test_dataset = SessionDataset(test_graphs, test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = EPCDModel(config)
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, config['train']['epochs'])

    # 加载最佳模型测试
    model.load_state_dict(torch.load('best_epcd_model.pt'))
    evaluator = Evaluator(model, config)
    metrics = evaluator.evaluate(test_loader)
    print("Test Metrics:", metrics)

if __name__ == '__main__':
    main()