"""Configuration for MirLLM-Graph"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model hyperparameters"""
    # LLM Configuration
    llm_model_name: str = "armheb/DNA_bert_6"  # Much smaller, ~110M params
    llm_hidden_size: int = 768
    freeze_llm: bool = True  # Freeze LLM to speed up training on CPU
    
    # GNN Configuration
    gcn_hidden_dim: int = 128
    gcn_output_dim: int = 128
    gcn_num_layers: int = 2
    gcn_dropout: float = 0.2
    
    # Fusion & Classification
    fusion_dim: int = 256
    num_classes: int = 2  # Binary: miRNA vs non-miRNA
    
    # Training
    llm_lr: float = 2e-5
    gnn_lr: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Data
    max_seq_length: int = 128  # Shorter sequences for CPU training
    
    # Device
    device: str = "cuda"