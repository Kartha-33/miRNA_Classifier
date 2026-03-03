"""
Advanced data augmentation for miRNA classification
"""

import random
from typing import List, Tuple
import numpy as np


class RNADataAugmenter:
    """Data augmentation strategies for RNA sequences"""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
        self.nucleotides = ['A', 'C', 'G', 'U']
    
    def random_mutation(self, sequence: str) -> str:
        """Random point mutations"""
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if random.random() < self.mutation_rate:
                seq_list[i] = random.choice(self.nucleotides)
        return ''.join(seq_list)
    
    def reverse_complement(self, sequence: str) -> str:
        """Reverse complement transformation"""
        complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement.get(n, 'N') for n in reversed(sequence))
    
    def subsequence_extraction(self, sequence: str, min_len: int = 18) -> str:
        """Extract random subsequence"""
        if len(sequence) <= min_len:
            return sequence
        start = random.randint(0, len(sequence) - min_len)
        length = random.randint(min_len, len(sequence) - start)
        return sequence[start:start+length]
    
    def augment_batch(self, sequences: List[str], labels: List[int],
                     augmentation_factor: int = 2) -> Tuple[List[str], List[int]]:
        """Apply augmentation to balance dataset"""
        aug_sequences, aug_labels = [], []
        
        for seq, label in zip(sequences, labels):
            aug_sequences.append(seq)
            aug_labels.append(label)
            
            for _ in range(augmentation_factor - 1):
                aug_type = random.choice(['mutation', 'reverse', 'subsequence'])
                
                if aug_type == 'mutation':
                    aug_seq = self.random_mutation(seq)
                elif aug_type == 'reverse':
                    aug_seq = self.reverse_complement(seq)
                else:
                    aug_seq = self.subsequence_extraction(seq)
                
                aug_sequences.append(aug_seq)
                aug_labels.append(label)
        
        return aug_sequences, aug_labels