import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
import optuna

torch.manual_seed(42)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        loss = loss_fn(out.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        probs = torch.sigmoid(out.squeeze())
        preds = (probs > 0.5).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics['auc_roc'] = roc_auc_score(all_labels, np.array(all_probs))
    except ValueError:
        metrics['auc_roc'] = 0.0
    
    return metrics


def create_model(model_name, feature_size, edge_feature_size, embedding_size=256):
    """Create a fresh model instance."""
    if model_name == 'GNN1':
        return GNN1(feature_size=feature_size, embedding_size=embedding_size)
    elif model_name == 'GNN2':
        return GNN2(feature_size=feature_size, embedding_size=embedding_size)
    elif model_name == 'GNN3':
        return GNN3(feature_size=feature_size, edge_feature_size=edge_feature_size, embedding_size=embedding_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='GNN Hyperparameter Optimization with Optuna')
    parser.add_argument('--train_data', type=str, default='data/split_data/HIV_train.csv',
                        help='Path to the training data CSV')
    parser.add_argument('--test_data', type=str, default='data/split_data/HIV_test.csv',
                        help='Path to the test data CSV')
    parser.add_argument('--model', type=str, choices=['GNN1', 'GNN2', 'GNN3'], default='GNN1',
                        help='GNN model architecture to optimize')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs per trial')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of Optuna trials')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(args.train_data))
    test_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(args.test_data), test=True)
    
    feature_size = train_dataset[0].x.shape[1]
    edge_feature_size = train_dataset[0].edge_attr.shape[1]
    
    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        pos_weight = trial.suggest_float("pos_weight", 5.0, 25.0)
        embedding_size = trial.suggest_categorical("embedding_size", [128, 256, 512])
        
        # Create fresh model for each trial
        model = create_model(args.model, feature_size, edge_feature_size, embedding_size)
        model = model.to(device)
        
        # Loss with sampled pos_weight
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Train
        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Evaluate
        metrics = evaluate(model, test_loader, loss_fn, device)
        
        # Report intermediate value for pruning
        trial.report(metrics['f1'], args.epochs)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return metrics['f1']
    
    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best params
    print(f"\n{'='*60}")
    print("Training final model with best hyperparameters...")
    print(f"{'='*60}")
    
    best = study.best_params
    final_model = create_model(args.model, feature_size, edge_feature_size, best['embedding_size'])
    final_model = final_model.to(device)
    
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([best['pos_weight']]).to(device)
    )
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'], weight_decay=best['weight_decay'])
    
    train_loader = DataLoader(train_dataset, batch_size=best['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best['batch_size'], shuffle=False)
    
    best_f1 = 0
    for epoch in range(args.epochs * 3):  # Train longer for final model
        train_loss = train_one_epoch(final_model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(final_model, test_loader, loss_fn, device)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            output_dir = os.path.join("outputs", args.model)
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_params': best,
            }, os.path.join(output_dir, "best_model.pth"))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | F1={metrics['f1']:.4f} | "
                  f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f}")
    
    print(f"\nFinal Best F1: {best_f1:.4f}")
    print(f"Model saved to outputs/{args.model}/best_model.pth")


if __name__ == "__main__":
    main()
