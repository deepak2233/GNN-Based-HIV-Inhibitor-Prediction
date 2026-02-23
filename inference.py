import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import argparse


def run_inference(model_type='GNN1', weights_path=None, test_data_path='data/split_data/HIV_test.csv',
                  batch_size=128):
    """Run inference on the test dataset using a trained GNN model."""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the test dataset
    test_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(test_data_path), test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get feature sizes from data
    feature_size = test_dataset[0].x.shape[1]
    edge_feature_size = test_dataset[0].edge_attr.shape[1]
    
    # Initialize model
    if model_type == 'GNN1':
        model = GNN1(feature_size=feature_size)
    elif model_type == 'GNN2':
        model = GNN2(feature_size=feature_size)
    elif model_type == 'GNN3':
        model = GNN3(feature_size=feature_size, edge_feature_size=edge_feature_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if weights_path is None:
        weights_path = os.path.join("outputs", model_type, "best_model.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Perform inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
            
            # Apply sigmoid for probabilities (model outputs raw logits)
            probs = torch.sigmoid(out.squeeze())
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*50}")
    print(f"Inference Results — {model_type}")
    print(f"{'='*50}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy:  {accuracy:.4f}")
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC AUC:   {roc_auc:.4f}")
    except ValueError:
        print("ROC AUC:   not defined (single class in labels)")
    
    return all_preds, all_labels, all_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN HIV Inference')
    parser.add_argument('--model_type', type=str, choices=['GNN1', 'GNN2', 'GNN3'], default='GNN1')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
    parser.add_argument('--test_data', type=str, default='data/split_data/HIV_test.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    run_inference(args.model_type, args.weights, args.test_data, args.batch_size)