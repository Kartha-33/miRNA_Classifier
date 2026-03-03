"""
Graph construction for RNA sequences
"""

import torch
from torch_geometric.data import Data
from typing import Optional, List, Tuple
from .structure_utils import parse_structure, validate_structure


def sequence_to_graph(
    sequence: str,
    structure: Optional[str] = None,
    add_self_loops: bool = True
) -> Data:
    """
    Convert RNA sequence to PyG graph
    
    Args:
        sequence: RNA sequence (ACGU)
        structure: Dot-bracket structure (optional)
        add_self_loops: Add self-loop edges
    
    Returns:
        PyG Data object with graph
    """
    # Validate inputs
    if structure is not None:
        structure = structure.strip()
        if len(structure) != len(sequence):
            # Try to fix common issues
            if len(structure) > len(sequence):
                structure = structure[:len(sequence)]
            else:
                # Pad with dots
                structure = structure + '.' * (len(sequence) - len(structure))
        
        # Validate structure
        if not validate_structure(sequence, structure):
            print(f"⚠️  Invalid structure, using backbone-only graph")
            structure = None
    
    seq_len = len(sequence)
    
    # Node features (one-hot encoding)
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}
    node_features = []
    
    for nuc in sequence.upper():
        feat = [0.0] * 5
        idx = nucleotide_map.get(nuc, 4)
        feat[idx] = 1.0
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge construction
    edge_list = []
    edge_types = []
    
    # 1. Backbone edges (sequential connections)
    for i in range(seq_len - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
        edge_types.extend([0, 0])  # 0 = backbone
    
    # 2. Base-pairing edges (from structure if available)
    if structure:
        pairs = parse_structure(structure)
        for i, j in pairs:
            if 0 <= i < seq_len and 0 <= j < seq_len:
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_types.extend([1, 1])  # 1 = base-pair
    
    # 3. Self-loops (optional)
    if add_self_loops:
        for i in range(seq_len):
            edge_list.append([i, i])
            edge_types.append(2)  # 2 = self-loop
    
    # Convert to tensor
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(seq_len)]
        edge_types = [2] * seq_len
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_types, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)