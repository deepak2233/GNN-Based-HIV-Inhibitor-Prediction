import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap

class GNN1(torch.nn.Module):
    def __init__(self, feature_size, embedding_size=256):
        super(GNN1, self).__init__()
        
        # GNN1 layers — 3x GAT + TopKPooling blocks
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.bn1 = BatchNorm1d(embedding_size * 3)
        self.head_transform1 = Linear(embedding_size * 3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.bn2 = BatchNorm1d(embedding_size * 3)
        self.head_transform2 = Linear(embedding_size * 3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.bn3 = BatchNorm1d(embedding_size * 3)
        self.head_transform3 = Linear(embedding_size * 3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.3)
        
        # Output MLP (takes concatenated multi-scale representations)
        self.linear1 = Linear(embedding_size * 3, 512)
        self.linear_bn = BatchNorm1d(512)
        self.linear2 = Linear(512, 1)
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        # First block
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.head_transform1(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool1(x, edge_index, None, batch_index)
        x1 = gap(x, batch_index)
        
        # Second block
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.head_transform2(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool2(x, edge_index, None, batch_index)
        x2 = gap(x, batch_index)

        # Third block
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.head_transform3(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, edge_index, None, batch_index)
        x3 = gap(x, batch_index)
        
        # Concat multi-scale pooled vectors (preserves more info than sum)
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Output block
        x = self.linear1(x)
        x = self.linear_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        
        return x