"""
Dataset loader for miRNA classification
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, Optional
from .graph_builder import sequence_to_graph
from .structure_utils import predict_structure_rnafold


class MiRNADataset(Dataset):
    """
    PyTorch Dataset for miRNA sequences
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "armheb/DNA_bert_6",
        max_length: int = 128,
        predict_structure: bool = False
    ):
        """
        Args:
            data_path: Path to CSV file with columns: sequence, label, [structure]
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length for tokenizer
            predict_structure: Auto-predict structures if missing (requires RNAfold)
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.graph_cache = {}
        
        # Check for structure column
        if 'structure' not in self.data.columns:
            print("⚠️  'structure' column not found.")
            if predict_structure:
                print("Attempting to predict structures with RNAfold...")
                self._predict_structures()
            else:
                print("Graphs will only have backbone edges.")
                self.data['structure'] = None
        
        # Fill NaN structures
        if 'structure' in self.data.columns:
            self.data['structure'] = self.data['structure'].fillna('')
        else:
            self.data['structure'] = ''
        
        # Print dataset statistics
        print(f"✓ Loaded {len(self.data)} samples")
        if 'label' in self.data.columns:
            label_counts = self.data['label'].value_counts()
            for label, count in label_counts.items():
                label_name = "miRNA" if label == 1 else "non-miRNA"
                print(f"  - {label_name}: {count}")
    
    def _predict_structures(self):
        """Predict missing structures using RNAfold"""
        from tqdm import tqdm
        
        structures = []
        for seq in tqdm(self.data['sequence'], desc="Predicting structures"):
            struct = predict_structure_rnafold(seq)
            if struct is None:
                struct = '.' * len(seq)
            structures.append(struct)
        
        self.data['structure'] = structures
        print("✓ Structures predicted")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary with:
                - input_ids: Tokenized sequence (Tensor)
                - attention_mask: Attention mask (Tensor)
                - graph: PyG graph object
                - label: Classification label (Tensor)
        """
        row = self.data.iloc[idx]
        
        sequence = row['sequence']
        structure = row.get('structure', None)
        if structure == '' or pd.isna(structure):
            structure = None
        
        label = row['label'] if 'label' in row else 0
        
        # Tokenize sequence
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Build graph (use cache)
        cache_key = f"{sequence}_{structure}"
        if cache_key not in self.graph_cache:
            graph = sequence_to_graph(sequence, structure)
            self.graph_cache[cache_key] = graph
        else:
            graph = self.graph_cache[cache_key]
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'graph': graph,
            'label': torch.tensor(label, dtype=torch.long)
        }
