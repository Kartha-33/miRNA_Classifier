# MirLLM-Graph

**Hybrid Multi-Modal Classifier for miRNA Detection**

A research-grade deep learning project combining **Genomic Language Models** (DNABERT-2) with **Geometric Deep Learning** (GCN) to distinguish miRNA from coding and non-coding sequences.

---

## 🧬 Architecture

```
             ┌─────────────────┐
             │  DNA Sequence   │
             └────────┬────────┘
                      │
             ┌────────┴────────┐
             │                 │
   ┌─────────▼─────────┐  ┌───▼──────────┐
   │   DNABERT-2-117M  │  │  GCN (PyG)   │
   │  (Sequence LLM)   │  │  (Structure) │
   └─────────┬─────────┘  └───┬──────────┘
             │                │
        [CLS] Token      Global Mean Pool
             │                │
             └────────┬───────┘
                      │
               ┌──────▼──────┐
               │   Fusion    │
               │     MLP     │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │  Binary     │
               │ Prediction  │
               └─────────────┘
```

### Key Components

1. **Sequence Branch**
   - DNABERT-2 (117M parameters)
   - Extracts semantic embeddings from DNA sequences
   - Uses [CLS] token representation (768-dim)

2. **Structure Branch**
   - Graph Convolutional Network (GCN)
   - Nodes: One-hot encoded nucleotides [A, C, G, T/U]
   - Edges: Backbone (sequential) + Hydrogen bonds (from secondary structure)
   - Global mean pooling (128-dim)

3. **Fusion & Classification**
   - Concatenate LLM + GNN embeddings
   - MLP classifier for binary prediction

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Create conda environment
conda create -n mirllm python=3.9
conda activate mirllm

# Install dependencies
pip install -r requirements.txt

# Optional: Install ViennaRNA for structure prediction
conda install -c bioconda viennarna
```

### Verify Installation

```bash
python -c "import torch; import torch_geometric; print('✓ Installation successful')"
```

---

## 📊 Data Requirements

### Data Format

Your CSV file should contain:

| Column      | Type   | Description                   | Required      |
| ----------- | ------ | ----------------------------- | ------------- |
| `sequence`  | string | DNA/RNA sequence (ACGT/U)     | ✅ Yes        |
| `structure` | string | Dot-bracket notation `((..))` | ⚠️ Optional\* |
| `label`     | int    | 0 = non-miRNA, 1 = miRNA      | ✅ Yes        |

\*If `structure` is missing, it will be predicted using RNAfold (requires ViennaRNA).

**Example CSV:**

```csv
sequence,structure,label
UGAGGUAGUAGGUUGUAUAGUU,(((((......)))))...,1
ACGTACGTACGTACGT,(((.......))).,0
```

### Data Sources

#### Positive Samples (miRNA)

- **miRBase** (Primary): https://www.mirbase.org/download/
  - `mature.fa`: ~40,000 mature miRNA sequences
  - `hairpin.fa`: Precursor miRNA sequences

#### Negative Samples (Non-miRNA)

- **RNAcentral**: https://rnacentral.org/ (tRNA, rRNA, lncRNA)
- **Ensembl**: http://ftp.ensembl.org/ (Coding sequences)
- **GENCODE**: https://www.gencodegenes.org/ (Non-coding annotations)

---

## 🚀 Quick Start

### Option 1: Automated Setup

```bash
chmod +x quick_start.sh
./quick_start.sh
```

This will:

1. Install dependencies
2. Download miRBase data
3. Generate negative samples
4. Verify installation

### Option 2: Manual Setup

#### Step 1: Prepare Data

```bash
# Download and prepare dataset
python download_data.py
```

This creates `data/mirna_dataset.csv` with:

- 5,000 positive samples (miRNA from miRBase)
- 5,000 negative samples (synthetic non-miRNA)

**Or use your own data:**

```python
import pandas as pd

df = pd.DataFrame({
    'sequence': ['UGAGGUAGUAGGUUGUAUAGUU', 'ACGTACGTACGT'],
    'structure': ['(((((......)))))...', '(((.....)))'],  # Optional
    'label': [1, 0]  # 1=miRNA, 0=non-miRNA
})
df.to_csv('data/my_dataset.csv', index=False)
```

#### Step 2: Test Graph Builder

```python
from src.graph_builder import sequence_to_graph

seq = "ACGTACGTACGT"
struct = "(((...)))..."
graph = sequence_to_graph(seq, struct)

print(f"Nodes: {graph.num_nodes}")        # 12
print(f"Edges: {graph.edge_index.size(1)}")  # 28 (backbone + H-bonds)
```

#### Step 3: Train Model

```bash
# Basic training
python -m src.train \
    --data_path data/mirna_dataset.csv \
    --output_dir outputs/ \
    --num_epochs 10

# With structure prediction (if structures missing)
python -m src.train \
    --data_path data/mirna_dataset.csv \
    --predict_structure \
    --output_dir outputs/ \
    --num_epochs 10

# GPU training with custom settings
python -m src.train \
    --data_path data/mirna_dataset.csv \
    --output_dir outputs/ \
    --batch_size 32 \
    --num_epochs 20 \
    --device cuda
```

**Training Outputs:**

- `outputs/best_model.pt` - Best model checkpoint (based on validation F1)
- `outputs/results.json` - Evaluation metrics & training history

---

## 📁 Project Structure

```
MirLLM-Graph/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py               # Configuration & hyperparameters
│   ├── structure_utils.py      # Dot-bracket parsing & RNAfold
│   ├── graph_builder.py        # PyG graph construction
│   ├── model.py                # HybridMirNA model
│   ├── dataset.py              # Custom dataset & collate
│   └── train.py                # Training loop
├── data/
│   └── mirna_dataset.csv       # Training data
├── download_data.py            # Data preparation script
├── quick_start.sh              # Automated setup script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🔬 Technical Details

### Graph Construction

**Nodes:** One-hot encoded nucleotides (4-dimensional)

- A: `[1, 0, 0, 0]`
- C: `[0, 1, 0, 0]`
- G: `[0, 0, 1, 0]`
- T/U: `[0, 0, 0, 1]`

**Edges:**

1. **Backbone Edges:** Sequential connectivity $i \leftrightarrow i+1$ (bidirectional)
2. **Hydrogen Bond Edges:** Parsed from dot-bracket notation
   - `(` at position $i$ paired with `)` at position $j$ → edge $(i, j)$

**Example:**

```python
sequence = "ACGT"
structure = "(())"

# Backbone edges: (0,1), (1,0), (1,2), (2,1), (2,3), (3,2)
# H-bond edges: (0,3), (3,0), (1,2), (2,1)
```

### Model Hyperparameters

```python
# LLM Configuration
llm_model_name = "zhihan1996/DNABERT-2-117M"
llm_hidden_size = 768
llm_lr = 2e-5

# GNN Configuration
gcn_hidden_dim = 128
gcn_output_dim = 128
gnn_lr = 1e-3

# Training
batch_size = 16
num_epochs = 10
max_seq_length = 512
```

### Training Strategy

- **Loss:** CrossEntropyLoss
- **Optimizer:** AdamW with differential learning rates
  - LLM parameters: 2e-5 (fine-tuning)
  - GNN parameters: 1e-3 (training from scratch)
- **Scheduler:** Linear warmup with decay
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC

---

## 🧪 Usage Examples

### Load Pretrained Model

```python
import torch
from src.model import HybridMirNA
from src.config import ModelConfig

config = ModelConfig()
model = HybridMirNA(config)

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best Val F1: {checkpoint['val_f1']:.4f}")
```

### Predict on New Sequences

```python
from src.graph_builder import sequence_to_graph
from torch_geometric.data import Batch

# Prepare input
seq = "UGAGGUAGUAGGUUGUAUAGUU"
struct = "(((((......)))))..."

# Tokenize
encoding = model.tokenizer(seq, return_tensors='pt', padding='max_length',
                          max_length=512, truncation=True)

# Build graph
graph = sequence_to_graph(seq, struct)
batched_graph = Batch.from_data_list([graph])

# Predict
with torch.no_grad():
    logits = model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        graph_x=batched_graph.x,
        graph_edge_index=batched_graph.edge_index,
        graph_batch=batched_graph.batch
    )

    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()

print(f"Prediction: {'miRNA' if pred == 1 else 'non-miRNA'}")
print(f"Confidence: {probs[0][pred]:.2%}")
```

---

## 📈 Expected Performance

With proper training data:

- **Accuracy:** 85-95%
- **F1 Score:** 84-93%
- **ROC-AUC:** 90-97%

_Performance depends on dataset quality, size, and class balance._

---

## 🛠️ Troubleshooting

### Issue: "RNAfold not found"

```bash
conda install -c bioconda viennarna
```

### Issue: "CUDA out of memory"

Reduce batch size:

```bash
python -m src.train --batch_size 8 --data_path data/mirna_dataset.csv
```

### Issue: "Tokenizer not found"

Check internet connection. The tokenizer downloads automatically from Hugging Face.

### Issue: "Graph has no edges"

Ensure structure strings match sequence lengths. Use `--predict_structure` flag.

---

## 📚 Dependencies

Core requirements:

- `torch >= 2.0.0`
- `torch-geometric >= 2.3.0`
- `transformers >= 4.30.0`
- `biopython >= 1.81`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`

See [requirements.txt](requirements.txt) for complete list.

---

## 🤝 Contributing

This is a research project. Contributions welcome!

Areas for improvement:

- Alternative GNN architectures (GAT, GIN)
- Attention-based fusion mechanisms
- Multi-task learning (pri-miRNA vs mature miRNA)
- Cross-species generalization

---

## 📝 Citation

_Research in progress_

If you use this code, please cite:

```
MirLLM-Graph: Hybrid Multi-Modal Classification of miRNA using
Geometric Deep Learning and Genomic Language Models (2026)
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 References

- **DNABERT-2:** https://github.com/Zhihan1996/DNABERT_2
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io/
- **miRBase:** https://www.mirbase.org/
- **ViennaRNA:** https://www.tbi.univie.ac.at/RNA/

---

**Built with ❤️ for RNA research and deep learning**
