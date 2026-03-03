"""
Installation verification tests for MirLLM-Graph
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required packages are installed"""
    try:
        import torch
        import torch_geometric
        import transformers
        import pandas
        import numpy
        import sklearn
        import plotly
        import networkx
        
        print("✓ All packages imported successfully")
        return True, "All imports successful"
    except ImportError as e:
        return False, f"Import failed: {str(e)}"


def test_device():
    """Test device availability"""
    try:
        import torch
        
        if torch.backends.mps.is_available():
            device = "mps"
            print("✓ MPS (Apple Silicon) available")
        elif torch.cuda.is_available():
            device = "cuda"
            print("✓ CUDA available")
        else:
            device = "cpu"
            print("✓ Using CPU")
        
        return True, f"Device: {device}"
    except Exception as e:
        return False, f"Device detection failed: {str(e)}"


def test_graph_builder():
    """Test graph construction with structures"""
    try:
        from src.graph_builder import sequence_to_graph
        
        sequence = "UGAGGUAGUAGGUUGUAUAGUU"
        structure = "(((((((.....)))))))"
        
        # Test with structure
        graph = sequence_to_graph(sequence, structure)
        
        # Validate graph
        assert graph.x.size(0) == len(sequence), "Node count mismatch"
        assert graph.edge_index.size(1) > 0, "No edges found"
        
        # Check for base-pairing edges
        edge_types = graph.edge_attr.unique().tolist()
        if len(edge_types) > 1:
            print("✓ Graph has backbone + base-pairing edges")
        else:
            print("⚠️  Graph has only backbone edges (structure may be missing)")
        
        return True, "Graph builder works"
        
    except Exception as e:
        return False, f"Graph builder failed: {str(e)}"


def test_structure_utils():
    """Test structure parsing"""
    try:
        from src.structure_utils import parse_structure
        
        structure = "(((...)))"
        pairs = parse_structure(structure)
        
        assert len(pairs) > 0, "No base pairs found"
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs), "Invalid pair format"
        
        print(f"✓ Parsed {len(pairs)} base pairs from structure")
        return True, "Structure parsing works"
        
    except Exception as e:
        return False, f"Structure parsing failed: {str(e)}"


def test_model():
    """Test model initialization"""
    try:
        from src.config import ModelConfig
        from src.model import HybridMirNA
        import torch
        
        config = ModelConfig()
        model = HybridMirNA(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model initialized successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True, "Model initialization successful"
        
    except Exception as e:
        return False, f"Model initialization failed: {str(e)}"


def test_dataset():
    """Test dataset loading"""
    try:
        from src.dataset import MiRNADataset
        import os
        
        data_path = "data/mirna_dataset.csv"
        
        if not os.path.exists(data_path):
            print(f"⚠️  Dataset not found at {data_path}")
            print("   Run: python download_data.py")
            return True, "Dataset test skipped (no data file)"
        
        print(f"Loading dataset from {data_path}...")
        dataset = MiRNADataset(data_path, predict_structure=False)
        
        print(f"✓ Loaded {len(dataset)} samples")
        
        # Check if structures exist
        import pandas as pd
        df = pd.read_csv(data_path)
        if 'structure' not in df.columns or df['structure'].isna().any():
            print("⚠️  WARNING: Some sequences missing structures!")
            print("   Run: python generate_structures.py")
            print("   This is required for full graph visualization!")
            return False, "Structures missing - run generate_structures.py"
        
        # Get sample
        sample = dataset[0]
        
        print(f"\n✓ Dataset loaded")
        print(f"  Total samples: {len(dataset)}")
        print(f"\nSample structure:")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  graph nodes: {sample['graph'].x.size(0)}")
        print(f"  graph edges: {sample['graph'].edge_index.size(1)}")
        print(f"  label: {sample['label']}")
        
        return True, "Dataset works"
        
    except Exception as e:
        return False, f"Dataset loading failed: {str(e)}"


def run_tests():
    """Run all tests"""
    tests = [
        ("Imports", test_imports),
        ("Device", test_device),
        ("Graph Builder", test_graph_builder),
        ("Structure Utils", test_structure_utils),
        ("Model", test_model),
        ("Dataset", test_dataset),
    ]
    
    results = []
    
    print("=" * 60)
    print("MirLLM-Graph Installation Tests")
    print("=" * 60)
    print()
    
    for name, test_func in tests:
        print("=" * 60)
        print(f"Testing {name}")
        print("=" * 60)
        
        try:
            passed, message = test_func()
            results.append((name, passed, message))
        except Exception as e:
            results.append((name, False, str(e)))
        
        print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed, message in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
            print(f"    {message}")
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✓ All tests passed! You're ready to train.")
        print()
        print("Next steps:")
        print("  1. Visualize samples: python visualize_interactive.py --data_path data/mirna_dataset.csv")
        print("  2. Train model: python -m src.train --data_path data/mirna_dataset.csv")
    else:
        print("✗ Some tests failed. Please fix issues before training.")
        print()
        print("Common fixes:")
        print("  - Missing structures: python generate_structures.py")
        print("  - Missing data: python download_data.py")
        print("  - Missing packages: pip install -r requirements.txt")
    
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
