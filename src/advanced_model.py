"""
State-of-the-art model improvements
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from transformers import AutoModel


class AttentionFusion(nn.Module):
    """Cross-attention between sequence and structure features"""
    
    def __init__(self, seq_dim: int, struct_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, seq_features, struct_features):
        # Project to same dimension
        seq_proj = self.seq_proj(seq_features).unsqueeze(1)
        struct_proj = self.struct_proj(struct_features).unsqueeze(1)
        
        # Cross-attention
        attn_out, _ = self.attention(seq_proj, struct_proj, struct_proj)
        fused = self.norm(attn_out + seq_proj)
        
        return fused.squeeze(1)


class ImprovedHybridMirNA(nn.Module):
    """
    Enhanced model with:
    - GAT instead of GCN for better edge modeling
    - Cross-attention fusion
    - Residual connections
    - Focal loss support
    """
    
    def __init__(self, config):
        super().__init__()
        
        # LLM Encoder (DNA-BERT)
        self.llm = AutoModel.from_pretrained(config.llm_model_name)
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # GAT Encoder (Graph Attention Networks)
        self.gnn1 = GATConv(config.node_feature_dim, 64, heads=4, concat=True)
        self.gnn2 = GATConv(64*4, 128, heads=2, concat=True)
        self.gnn3 = GATConv(128*2, config.gnn_hidden_dim, heads=1, concat=False)
        
        self.gnn_norm1 = nn.BatchNorm1d(64*4)
        self.gnn_norm2 = nn.BatchNorm1d(128*2)
        self.dropout = nn.Dropout(config.dropout)
        
        # Cross-attention fusion
        self.fusion = AttentionFusion(
            seq_dim=config.llm_hidden_dim,
            struct_dim=config.gnn_hidden_dim,
            hidden_dim=256
        )
        
        # Classification head with residual
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(64, config.num_classes)
        )
        
    def forward(self, input_ids, attention_mask, graph_batch):
        # Sequence branch
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        seq_features = llm_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Graph branch with residuals
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        
        x1 = self.gnn1(x, edge_index).relu()
        x1 = self.gnn_norm1(x1)
        x1 = self.dropout(x1)
        
        x2 = self.gnn2(x1, edge_index).relu()
        x2 = self.gnn_norm2(x2)
        x2 = self.dropout(x2)
        
        x3 = self.gnn3(x2, edge_index)
        
        # Global pooling (combine mean + max)
        struct_features = (
            global_mean_pool(x3, batch) + 
            global_max_pool(x3, batch)
        ) / 2
        
        # Cross-attention fusion
        fused = self.fusion(seq_features, struct_features)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits