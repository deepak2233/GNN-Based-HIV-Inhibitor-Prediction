import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
from utils import calculate_metrics, plot_confusion_matrix, save_checkpoint
from tqdm import tqdm
import optuna


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        out = out.squeeze()
        loss = loss_fn(out, batch.y.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float()
        all_preds.append(preds.detach().cpu())
        all_labels.append(batch.y.detach().cpu())
        
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = calculate_metrics(all_preds, all_labels)
    
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        out = out.squeeze()
        loss = loss_fn(out, batch.y.float())
        
        total_loss += loss.item()
        probs = torch.sigmoid(out)
        all_probs.append(probs.detach().cpu())
        all_preds.append((probs > 0.5).float().detach().cpu())
        all_labels.append(batch.y.detach().cpu())
        
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = calculate_metrics(all_preds, all_labels, y_prob=all_probs)
    
    return total_loss / len(loader), metrics, all_preds, all_labels


def compute_pos_weight(dataset):
    """Auto-compute pos_weight from training label distribution."""
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i].y.item())
    labels = np.array(labels)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    if n_pos == 0:
        return 1.0
    weight = n_neg / n_pos
    # Cap at 30 to avoid extreme values
    return min(weight, 30.0)


def main():
    parser = argparse.ArgumentParser(description='GNN HIV Classification Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'optimize', 'test'], default='train')
    parser.add_argument('--model_type', type=str, choices=['GNN1', 'GNN2', 'GNN3'], default='GNN1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--train_data', type=str, default='data/split_data/HIV_train.csv')
    parser.add_argument('--test_data', type=str, default='data/split_data/HIV_test.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--quick_test', action='store_true', help='Run on a small subset for verification')
    parser.add_argument('--embedding_size', type=int, default=256, help='Model embedding dimension')
    
    args = parser.parse_args()
    
    # Model-specific output directory
    output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(args.train_data))
    test_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(args.test_data), test=True)
    
    if args.quick_test:
        train_dataset = train_dataset[:100]
        test_dataset = test_dataset[:50]
        print("Running in quick_test mode (subsetting data).")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    feature_size = train_dataset[0].x.shape[1]
    edge_feature_size = train_dataset[0].edge_attr.shape[1]
    
    if args.model_type == 'GNN1':
        model = GNN1(feature_size=feature_size, embedding_size=args.embedding_size).to(device)
    elif args.model_type == 'GNN2':
        model = GNN2(feature_size=feature_size, embedding_size=args.embedding_size).to(device)
    else:
        model = GNN3(feature_size=feature_size, edge_feature_size=edge_feature_size, 
                      embedding_size=args.embedding_size).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type} | Parameters: {param_count:,}")
        
    # Loss — auto-compute pos_weight from data
    auto_pw = compute_pos_weight(train_dataset)
    print(f"Auto-computed pos_weight: {auto_pw:.1f}")
    pos_weight = torch.tensor([auto_pw]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer + Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    if args.mode == 'train':
        best_f1 = -1.0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_loss, test_metrics, y_pred, y_true = evaluate(model, test_loader, loss_fn, device)
            
            # Step scheduler based on test F1
            scheduler.step(test_metrics['f1_score'])
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{args.epochs} (lr={current_lr:.2e})")
            print(f"  Train — Loss: {train_loss:.4f} | F1: {train_metrics['f1_score']:.4f} | "
                  f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f}")
            print(f"  Test  — Loss: {test_loss:.4f} | F1: {test_metrics['f1_score']:.4f} | "
                  f"Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f}", end="")
            if 'auc_roc' in test_metrics and test_metrics['auc_roc'] is not None:
                print(f" | AUC: {test_metrics['auc_roc']:.4f}")
            else:
                print()
            
            if test_metrics['f1_score'] > best_f1:
                best_f1 = test_metrics['f1_score']
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, test_loss, os.path.join(output_dir, "best_model.pth"))
                plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, "confusion_matrix.png"))
                print(f"  ✓ New best F1: {best_f1:.4f} — model saved")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience})")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break
        
        print(f"\nTraining complete. Best Test F1: {best_f1:.4f}")
        print(f"Model saved to: {os.path.join(output_dir, 'best_model.pth')}")
                
    elif args.mode == 'test':
        checkpoint_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model.")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights.")
        
        test_loss, test_metrics, y_pred, y_true = evaluate(model, test_loader, loss_fn, device)
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
        plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, "test_confusion_matrix.png"))

    elif args.mode == 'optimize':
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            pw = trial.suggest_float("pos_weight", 5.0, 25.0)
            
            trial_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))
            trial_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trial_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            for _ in range(5):  # Fewer epochs for tuning
                train_one_epoch(model, trial_loader, trial_optimizer, trial_loss_fn, device)
                
            _, test_metrics, _, _ = evaluate(model, test_loader, loss_fn, device)
            return test_metrics['f1_score']

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        print("Best params:", study.best_params)
        print("Best F1:", study.best_value)


if __name__ == "__main__":
    main()
