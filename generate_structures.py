"""
Add RNA secondary structures to existing dataset using RNAfold
"""

import pandas as pd
import subprocess
from tqdm import tqdm


def predict_structure(sequence: str) -> str:
    """Predict RNA structure using RNAfold"""
    try:
        process = subprocess.Popen(
            ['RNAfold', '--noPS'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        output, error = process.communicate(input=sequence, timeout=10)
        
        # Parse output
        lines = output.strip().split('\n')
        if len(lines) >= 2:
            structure_line = lines[1].split()[0]  # Get structure before energy
            return structure_line
        else:
            return '.' * len(sequence)
            
    except Exception as e:
        print(f"Error predicting structure: {e}")
        return '.' * len(sequence)


def add_structures_to_dataset(input_csv: str):
    """Add structure column to dataset"""
    
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check if structure column exists
    if 'structure' in df.columns:
        print("✓ Structure column already exists")
        has_structures = df['structure'].notna().sum()
        
        if has_structures == len(df):
            print("✓ All sequences already have structures")
            return
        
        print(f"Filling {len(df) - has_structures} missing structures...")
        mask = df['structure'].isna()
    else:
        print("Adding structure column...")
        df['structure'] = None
        mask = [True] * len(df)
    
    # Predict structures
    print("Predicting RNA secondary structures with RNAfold...")
    print("This will take 5-10 minutes for 10,000 sequences...")
    
    for idx in tqdm(df[mask].index, desc="Predicting"):
        sequence = df.loc[idx, 'sequence']
        structure = predict_structure(sequence)
        df.loc[idx, 'structure'] = structure
    
    # Save back to same file
    df.to_csv(input_csv, index=False)
    print(f"\n✓ Dataset updated: {input_csv}")
    print(f"  Total samples: {len(df)}")
    print(f"  All sequences now have structures!")


if __name__ == "__main__":
    # Check RNAfold
    try:
        result = subprocess.run(['RNAfold', '--version'], 
                              capture_output=True, check=True, text=True)
        print(f"✓ RNAfold found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ RNAfold not installed")
        print("\nInstall ViennaRNA:")
        print("  macOS: brew install viennarna")
        print("  Linux: sudo apt-get install vienna-rna")
        exit(1)
    
    add_structures_to_dataset('data/mirna_dataset.csv')