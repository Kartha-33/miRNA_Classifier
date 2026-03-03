#!/bin/bash

# Quick Start Script for MirLLM-Graph

echo "=========================================="
echo "MirLLM-Graph Quick Start"
echo "=========================================="
echo ""

# Step 1: Install dependencies
echo "[Step 1/4] Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Step 2: Download data
echo "[Step 2/4] Downloading and preparing data..."
python download_data.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to download data"
    exit 1
fi
echo "✓ Data prepared"
echo ""

# Step 3: Verify installation
echo "[Step 3/4] Verifying installation..."
python -c "
import torch
import torch_geometric
from transformers import AutoTokenizer
print('✓ PyTorch:', torch.__version__)
print('✓ PyG:', torch_geometric.__version__)
print('✓ Transformers available')
"

if [ $? -ne 0 ]; then
    echo "❌ Verification failed"
    exit 1
fi
echo ""

# Step 4: Test graph builder
echo "[Step 4/4] Testing graph builder..."
python -c "
from src.graph_builder import sequence_to_graph

seq = 'ACGTACGTACGT'
struct = '(((...)))...'
graph = sequence_to_graph(seq, struct)
print(f'✓ Graph built: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges')
"

if [ $? -ne 0 ]; then
    echo "❌ Graph builder test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  python -m src.train --data_path data/mirna_dataset.csv --output_dir outputs/"
echo ""
echo "Optional: Install ViennaRNA for structure prediction"
echo "  conda install -c bioconda viennarna"
echo ""
