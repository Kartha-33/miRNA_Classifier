"""
Hybrid Multi-Modal Model: DNA Transformer + GCN (Structure)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from .config import ModelConfig


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for secondary structure."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input: 4-dim one-hot encoding
        input_dim = 4
        hidden_dim = config.gcn_hidden_dim
        output_dim = config.gcn_output_dim
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(config.gcn_dropout)
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, 4]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level embedding [batch_size, output_dim]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)  # [batch_size, output_dim]
        
        return x


class HybridMirNA(nn.Module):
    """
    Hybrid Multi-Modal Classifier for miRNA detection.
    
    Architecture:
        - Branch 1: DNA Transformer (sequence semantics)
        - Branch 2: GCN (structural topology)
        - Fusion: Concatenate embeddings
        - Head: Binary classification MLP
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Branch 1: DNA Language Model
        print(f"Loading {config.llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        self.llm = AutoModel.from_pretrained(
            config.llm_model_name, 
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Branch 2: Graph Neural Network
        self.gnn = GNNEncoder(config)
        
        # Fusion dimensions
        fusion_input_dim = config.llm_hidden_size + config.gcn_output_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.fusion_dim, config.num_classes)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor
    ):
        """
        Forward pass through both branches and fusion.
        
        Args:
            input_ids: Tokenized sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            graph_x: Node features [total_nodes, 4]
            graph_edge_index: Edge connectivity [2, total_edges]
            graph_batch: Batch assignment [total_nodes]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Branch 1: LLM (Sequence)
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Extract [CLS] token embedding (first token)
        llm_emb = llm_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Branch 2: GNN (Structure)
        gnn_emb = self.gnn(graph_x, graph_edge_index, graph_batch)  # [batch_size, gcn_output_dim]
        
        # Fusion: Concatenate both embeddings
        fused = torch.cat([llm_emb, gnn_emb], dim=1)  # [batch_size, fusion_input_dim]
        
        # Classification
        logits = self.classifier(fused)  # [batch_size, num_classes]
        
        return logits
    
    def get_separate_parameter_groups(self):
        """
        Get parameter groups with different learning rates.
        
        Returns:
            List of dicts for optimizer parameter groups
        """
        llm_params = list(self.llm.parameters())
        gnn_params = list(self.gnn.parameters())
        classifier_params = list(self.classifier.parameters())
        
        return [
            {'params': llm_params, 'lr': self.config.llm_lr},
            {'params': gnn_params + classifier_params, 'lr': self.config.gnn_lr}
        ]
