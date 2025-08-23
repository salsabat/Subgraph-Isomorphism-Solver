import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple, List
import networkx as nx

class GraphTransformerMatcher(nn.Module):
    def __init__(self, max_nodes=32, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.max_nodes = max_nodes
        self.d_model = d_model
        
        self.node_embedding = nn.Linear(max_nodes, d_model)
        self.pos_embedding = nn.Embedding(max_nodes, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.subgraph_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(d_model, max_nodes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, subgraph_adj, target_adj, subgraph_mask, target_mask):
        batch_size = subgraph_adj.size(0)
        sub_n = subgraph_mask.sum(dim=1).max().item()
        target_n = target_mask.sum(dim=1).max().item()
        
        sub_emb = self.node_embedding(subgraph_adj[:, :sub_n, :])
        target_emb = self.node_embedding(target_adj[:, :target_n, :])
        
        pos_ids_sub = torch.arange(sub_n, device=subgraph_adj.device).unsqueeze(0).expand(batch_size, -1)
        pos_ids_target = torch.arange(target_n, device=target_adj.device).unsqueeze(0).expand(batch_size, -1)
        
        sub_emb = sub_emb + self.pos_embedding(pos_ids_sub)
        target_emb = target_emb + self.pos_embedding(pos_ids_target)
        
        sub_encoded = self.subgraph_encoder(sub_emb, src_key_padding_mask=~subgraph_mask[:, :sub_n])
        target_encoded = self.target_encoder(target_emb, src_key_padding_mask=~target_mask[:, :target_n])
        
        matched_features, attention_weights = self.cross_attention(
            query=sub_encoded,
            key=target_encoded, 
            value=target_encoded,
            key_padding_mask=~target_mask[:, :target_n]
        )
        
        matched_features = self.dropout(matched_features)
        
        logits = self.output_proj(matched_features)
        
        target_mask_expanded = target_mask.unsqueeze(1).expand(-1, sub_n, -1)
        logits = logits.masked_fill(~target_mask_expanded, float('-inf'))
        
        return logits[:, :sub_n, :target_n]

class GraphPairDataset(Dataset):
    def __init__(self, num_samples=10000, max_nodes=32):
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.data = []
        
        for _ in range(num_samples):
            sub_size = random.randint(3, min(8, max_nodes-2))
            target_size = random.randint(sub_size + 2, max_nodes)
            
            subgraph, target, mapping = self.generate_pair(sub_size, target_size)
            self.data.append((subgraph, target, mapping, sub_size, target_size))
    
    def generate_pair(self, sub_size, target_size):
        subgraph = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        
        for i in range(sub_size - 1):
            subgraph[i][i + 1] = 1.0
            subgraph[i + 1][i] = 1.0
        
        for _ in range(random.randint(0, sub_size)):
            u, v = random.sample(range(sub_size), 2)
            subgraph[u][v] = 1.0
            subgraph[v][u] = 1.0
        
        target = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        
        mapping = random.sample(range(target_size), sub_size)
        
        for u in range(sub_size):
            for v in range(sub_size):
                if subgraph[u][v] == 1.0:
                    target[mapping[u]][mapping[v]] = 1.0
        
        for _ in range(random.randint(target_size, target_size * 2)):
            u, v = random.sample(range(target_size), 2)
            target[u][v] = 1.0
            target[v][u] = 1.0
        
        true_mapping = np.full(self.max_nodes, -1, dtype=np.int64)
        for sub_node, target_node in enumerate(mapping):
            true_mapping[sub_node] = target_node
        
        return subgraph, target, true_mapping, mapping
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        subgraph, target, mapping, sub_size, target_size = self.data[idx]
        
        # Create masks
        sub_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        sub_mask[:sub_size] = True
        
        target_mask = torch.zeros(self.max_nodes, dtype=torch.bool) 
        target_mask[:target_size] = True
        
        return {
            'subgraph': torch.tensor(subgraph),
            'target': torch.tensor(target),
            'mapping': torch.tensor(mapping),
            'sub_mask': sub_mask,
            'target_mask': target_mask,
            'sub_size': sub_size,
            'target_size': target_size
        }

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GraphTransformerMatcher().to(device)
    dataset = GraphPairDataset(num_samples=20000)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    best_val_acc = 0.0
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            subgraph = batch['subgraph'].to(device)
            target = batch['target'].to(device) 
            mapping = batch['mapping'].to(device)
            sub_mask = batch['sub_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(subgraph, target, sub_mask, target_mask)
            
            loss = 0.0
            correct = 0
            total = 0
            
            for i in range(logits.size(0)):
                sub_size = batch['sub_size'][i].item()
                valid_mapping = mapping[i][:sub_size]
                valid_logits = logits[i][:sub_size]
                
                loss += F.cross_entropy(valid_logits, valid_mapping)
                
                pred = valid_logits.argmax(dim=1)
                correct += (pred == valid_mapping).sum().item()
                total += sub_size
            
            loss = loss / logits.size(0)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += correct
            train_total += total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                subgraph = batch['subgraph'].to(device)
                target = batch['target'].to(device)
                mapping = batch['mapping'].to(device) 
                sub_mask = batch['sub_mask'].to(device)
                target_mask = batch['target_mask'].to(device)
                
                logits = model(subgraph, target, sub_mask, target_mask)
                
                loss = 0.0
                correct = 0
                total = 0
                
                for i in range(logits.size(0)):
                    sub_size = batch['sub_size'][i].item()
                    valid_mapping = mapping[i][:sub_size]
                    valid_logits = logits[i][:sub_size]
                    
                    loss += F.cross_entropy(valid_logits, valid_mapping)
                    
                    pred = valid_logits.argmax(dim=1)
                    correct += (pred == valid_mapping).sum().item()
                    total += sub_size
                
                val_loss += loss.item() / logits.size(0)
                val_correct += correct
                val_total += total
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()

def export_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GraphTransformerMatcher().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    batch_size = 1
    max_nodes = 32
    
    dummy_subgraph = torch.randn(batch_size, max_nodes, max_nodes).to(device)
    dummy_target = torch.randn(batch_size, max_nodes, max_nodes).to(device)
    dummy_sub_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool).to(device)
    dummy_target_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool).to(device)
    
    torch.onnx.export(
        model,
        (dummy_subgraph, dummy_target, dummy_sub_mask, dummy_target_mask),
        "graph_matcher.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['subgraph', 'target', 'sub_mask', 'target_mask'],
        output_names=['logits'],
        dynamic_axes={
            'subgraph': {0: 'batch_size'},
            'target': {0: 'batch_size'}, 
            'sub_mask': {0: 'batch_size'},
            'target_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    train_model()
    export_to_onnx()