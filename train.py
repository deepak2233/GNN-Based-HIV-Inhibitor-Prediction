#%%
import os
import rdkit
import torch
import numpy as np
import torch_geometric
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem import rdBase
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
import torch.nn.functional as F 
import seaborn as sns
from torch.nn import Sequential, Linear, BatchNorm1d, ModuleList, ReLU
from torch_geometric.nn import TransformerConv, TopKPooling, GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
torch.manual_seed(42)

data_path = 'data/raw_data/HIV_data.csv'
data = pd.read_csv(data_path, index_col=[0])
print("###### Raw Data Shape - ")
print(data.shape)
print("0 : ", data["HIV_active"].value_counts()[0])
print("1 : ", data["HIV_active"].value_counts()[1])

#%%
# Split the data, Due to imbalance datasets we need to set train ration high
print("###### After Data split Shape - ")
train_data = pd.read_csv('data/split_data/HIV_train.csv')
test_data = pd.read_csv('data/split_data/HIV_test.csv')

print("Train Data ", train_data.shape)
print("0 : ", train_data["HIV_active"].value_counts()[0])
print("1 : ", train_data["HIV_active"].value_counts()[1])

print("Test Data ", test_data.shape)
print("0 : ", test_data["HIV_active"].value_counts()[0])
print("1 : ", test_data["HIV_active"].value_counts()[1])

if os.path.exists('data/split_data/HIV_train_oversampled.csv'):
    train_data_oversampled = pd.read_csv('data/split_data/HIV_train_oversampled.csv')
    print("Train Data Oversampled ", train_data_oversampled.shape)
    print("0 : ", train_data_oversampled["HIV_active"].value_counts()[0])
    print("1 : ", train_data_oversampled["HIV_active"].value_counts()[1])


#%%
print("######## Loading dataset...")
# Use raw train data with weighted loss (better than oversampling for GNNs)
train_dataset = MoleculeDataset(root="data/split_data", filename="HIV_train.csv")
test_dataset = MoleculeDataset(root="data/split_data", filename="HIV_test.csv", test=True)       

# %%
print("######## Loading GNN1 Model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GNN1(feature_size=train_dataset[0].x.shape[1])
model = model.to(device)
print(f"######### Number of parameters: {count_parameters(model)}")
print(model)

# %%
# Loss and Optimizer
# Use BCEWithLogitsLoss with pos_weight for class imbalance handling
# (models output raw logits, not probabilities)
pos_weight = torch.tensor([15.0]).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# %%
NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset,
                          batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset,
                          batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)

#%%


#%%

def train(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Using GPU
        batch.to(device)
        # Reset gradient
        optimizer.zero_grad()
        # Passing the node feature and edge info
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),  # Fixed: was .float (missing parentheses)
                     batch.edge_index,
                     batch.batch)
        
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step

#%%   
def test(epoch, model, test_loader, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        # Using GPU
        batch.to(device)
        pred = model(batch.x.float(),
                    batch.edge_attr.float(),  # Fixed: was .float (missing parentheses)
                    batch.edge_index,
                    batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())            
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss / step

#%%

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    os.makedirs('data/images', exist_ok=True)
    cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')
    plt.close()

   
def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
    except:
        print(f"ROC AUC: not defined")

# %%
# Start training
print("###### Start GNN Model training")
best_loss = 1000
early_stopping_counter = 0
for epoch in range(300): 
    if early_stopping_counter <= 10:  # = x * 5 
        # Training
        model.train()
        loss = train(epoch, model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch} | Train Loss {loss}")

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch, model, test_loader, loss_fn)
            print(f"Epoch {epoch} | Test Loss {loss}")
            
            # Update best loss
            if float(loss) < best_loss:
                best_loss = loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        scheduler.step()
    else:
        print("Early stopping due to no improvement.")
        print([best_loss])
        break

print(f"Finishing training with best test loss: {best_loss}")
print([best_loss])

output_folder = "model_weight"
os.makedirs(output_folder, exist_ok=True)
model_path = os.path.join(output_folder, "model.pth")  # Fixed: was os.join.path
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# %%
