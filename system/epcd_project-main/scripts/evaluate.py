import sys
sys.path.append('..')
import yaml
import torch
from src.data_loader import DataLoader as Preprocessor
from src.path_extractor import PathExtractor
from src.models.epcd_model import EPCDModel
from src.evaluator import Evaluator
from torch.utils.data import DataLoader, Dataset
from train import SessionDataset

def main():
    with open('../configs/epcd_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    preprocessor = Preprocessor(config)
    graph_sequences, labels = preprocessor.load_processed()
    path_extractor = PathExtractor(config)
    path_sequences = path_extractor.extract_all(graph_sequences)
    dataset = SessionDataset(graph_sequences, path_sequences, labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = EPCDModel(config)
    model.load_state_dict(torch.load('best_epcd_model.pt'))
    evaluator = Evaluator(model, config)
    metrics = evaluator.evaluate(loader)
    print("Test Metrics:", metrics)

if __name__ == '__main__':
    main()