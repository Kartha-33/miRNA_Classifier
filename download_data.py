"""
Script to download and prepare miRNA datasets for training.

Datasets:
1. miRBase: Mature miRNA sequences
2. RNAcentral: Non-coding RNA sequences (for negative samples)
3. Ensembl: Coding sequences (for negative samples)
"""

import os
import requests
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import subprocess
import warnings


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_mirbase():
    """
    Download mature miRNA sequences from miRBase.
    
    Source: https://www.mirbase.org/download/
    
    Returns mature miRNA sequences (positive samples).
    """
    print("\n[1/3] Downloading miRBase mature miRNA sequences...")
    
    url = "https://www.mirbase.org/download/mature.fa"
    output_path = DATA_DIR / "mirbase_mature.fa"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Count sequences
        count = sum(1 for _ in SeqIO.parse(output_path, "fasta"))
        print(f"✓ Downloaded {count} mature miRNA sequences")
        return output_path
        
    except Exception as e:
        print(f"✗ Failed to download miRBase: {e}")
        return None


def download_negative_samples():
    """
    Create negative samples from public RNA databases.
    
    Options:
    - RNAcentral non-miRNA sequences
    - Random non-coding sequences
    - Coding sequences (mRNA)
    """
    print("\n[2/3] Preparing negative samples...")
    
    # For demo purposes, we'll create synthetic negative samples
    # In production, use actual databases
    
    print("Note: Using synthetic negative samples for demonstration.")
    print("For production, download from:")
    print("  - RNAcentral: https://rnacentral.org/")
    print("  - Ensembl: http://ftp.ensembl.org/")
    
    negative_seqs = []
    
    # Generate random non-miRNA sequences
    import random
    bases = ['A', 'C', 'G', 'T']
    
    for i in range(5000):  # Create 5000 negative samples
        # Vary length (100-1000 bp, longer than typical miRNA)
        length = random.randint(100, 1000)
        seq = ''.join(random.choices(bases, k=length))
        negative_seqs.append({
            'id': f'NEG_{i:05d}',
            'sequence': seq,
            'label': 0
        })
    
    print(f"✓ Generated {len(negative_seqs)} negative samples")
    return negative_seqs


def predict_structures(sequences, max_sequences=1000):
    """
    Predict secondary structures using RNAfold.
    
    Args:
        sequences: List of sequence strings
        max_sequences: Maximum sequences to predict (for speed)
        
    Returns:
        List of dot-bracket structures
    """
    print("\n[3/3] Predicting secondary structures with RNAfold...")
    
    # Check if RNAfold is available
    try:
        subprocess.run(['RNAfold', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  RNAfold not found. Structures will be predicted during training.")
        print("   To install: conda install -c bioconda viennarna")
        return [None] * len(sequences)
    
    structures = []
    limit = min(len(sequences), max_sequences)
    
    print(f"   Predicting structures for {limit} sequences (this may take a while)...")
    
    from src.structure_utils import call_rnafold
    
    for i, seq in enumerate(sequences[:limit]):
        if i % 100 == 0:
            print(f"   Progress: {i}/{limit}")
        
        struct = call_rnafold(seq)
        structures.append(struct)
    
    # Fill remaining with None
    structures.extend([None] * (len(sequences) - limit))
    
    print(f"✓ Predicted {sum(s is not None for s in structures)} structures")
    return structures


def create_dataset():
    """Main function to create the training dataset."""
    print("="*60)
    print("MirLLM-Graph Data Preparation")
    print("="*60)
    
    # 1. Download positive samples (miRNA)
    mirbase_path = download_mirbase()
    
    if mirbase_path and mirbase_path.exists():
        positive_samples = []
        for record in SeqIO.parse(mirbase_path, "fasta"):
            positive_samples.append({
                'id': record.id,
                'sequence': str(record.seq),
                'label': 1
            })
    else:
        print("Using fallback miRNA sequences...")
        # Fallback: some real miRNA examples
        positive_samples = [
            {'id': 'hsa-let-7a', 'sequence': 'UGAGGUAGUAGGUUGUAUAGUU', 'label': 1},
            {'id': 'hsa-miR-21', 'sequence': 'UAGCUUAUCAGACUGAUGUUGA', 'label': 1},
            {'id': 'hsa-miR-155', 'sequence': 'UUAAUGCUAAUCGUGAUAGGGGU', 'label': 1},
        ]
    
    # Limit positive samples for demo
    positive_samples = positive_samples[:5000]
    print(f"\n✓ Collected {len(positive_samples)} positive samples (miRNA)")
    
    # 2. Get negative samples
    negative_samples = download_negative_samples()
    
    # 3. Combine
    all_samples = positive_samples + negative_samples
    print(f"\n✓ Total samples: {len(all_samples)}")
    print(f"   - Positive (miRNA): {len(positive_samples)}")
    print(f"   - Negative: {len(negative_samples)}")
    
    # 4. Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # 5. Predict structures (optional, comment out if slow)
    print("\nDo you want to predict structures now? (recommended for better performance)")
    print("This can take a while. You can skip and predict during training.")
    
    # For automation, skip structure prediction here
    # Uncomment to enable:
    # structures = predict_structures(df['sequence'].tolist())
    # df['structure'] = structures
    
    # 6. Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 7. Save
    output_file = DATA_DIR / "mirna_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"✓ Dataset saved to: {output_file}")
    print("="*60)
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Positive (miRNA): {(df['label'] == 1).sum()}")
    print(f"  Negative: {(df['label'] == 0).sum()}")
    print(f"  Columns: {list(df.columns)}")
    
    if 'structure' in df.columns:
        structures_present = df['structure'].notna().sum()
        print(f"  Structures available: {structures_present}/{len(df)}")
    
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Train model: python -m src.train --data_path data/mirna_dataset.csv")
    
    return output_file


if __name__ == '__main__':
    create_dataset()
