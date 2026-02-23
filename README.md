# 🧬 GNN-Based HIV Inhibitor Prediction

A **Graph Neural Network** pipeline for predicting HIV inhibitor activity from molecular structures. Built with **PyTorch Geometric**, featuring three progressively advanced GNN architectures, an optimized training pipeline, and a premium **Streamlit** web app for interactive drug discovery.

---

## Table of Contents

- [Project Overview](#project-overview)
- [End-to-End Architecture Flow](#end-to-end-architecture-flow)
- [GNN Model Architectures](#gnn-model-architectures)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [Streamlit Web App](#streamlit-web-app)
- [Dataset](#dataset)
- [Results & Metrics](#results--metrics)
- [Optimization Techniques](#optimization-techniques)
- [Tech Stack](#tech-stack)

---

## Project Overview

The **DTP AIDS Antiviral Screen** dataset contains ~41,000 molecules screened for HIV inhibition activity. This project builds GNN classifiers that learn directly from molecular graph structure to predict whether a compound is **HIV Active** or **Inactive**.

```
🎯 Goal: Given a molecule (SMILES string) → Predict HIV inhibitor activity (Active / Inactive)
```

### Key Challenges
- **Severe class imbalance**: Only ~3.5% of molecules are active (ratio 1:26)
- **Graph-structured data**: Molecules are not vectors — they are graphs with atoms (nodes) and bonds (edges)
- **Small active class**: Must maximize recall without destroying precision

---

## End-to-End Architecture Flow

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
  │  SMILES  │────▶│  Molecule    │────▶│  Graph      │────▶│  GNN     │
  │  String  │     │  Parsing     │     │  Features   │     │  Model   │
  └──────────┘     └──────────────┘     └─────────────┘     └────┬─────┘
                                                                  │
       ┌──────────────────────────────────────────────────────────┘
       ▼
  ┌──────────┐     ┌──────────────┐     ┌─────────────┐
  │  Raw     │────▶│  Sigmoid     │────▶│  Active /   │
  │  Logit   │     │  Threshold   │     │  Inactive   │
  └──────────┘     └──────────────┘     └─────────────┘
```

### Detailed Data Flow

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1: RAW DATA
  ━━━━━━━━━━━━━━━━
  HIV_data.csv (41,127 molecules)
       │
       ▼
  ┌─────────────────────────────┐
  │  Train/Test Split (80/20)   │
  │  HIV_train.csv (32,901)     │
  │  HIV_test.csv  (8,238)      │
  └─────────────┬───────────────┘
                │
                ▼
  STEP 2: MOLECULE → GRAPH CONVERSION (dataset_featurizer.py)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
  SMILES: "CC(=O)OC1=CC=CC=C1C(=O)O"
       │
       ▼ RDKit parsing
  ┌─────────────────────────────────────────────┐
  │              Molecular Graph                │
  │                                             │
  │    C ── C ── O          Node Features (9):  │
  │    ‖         │          ┌─────────────────┐ │
  │    O    ┌────┴────┐     │ Atomic Number   │ │
  │         │         │     │ Degree          │ │
  │    C == C    C == C     │ Formal Charge   │ │
  │    │         │          │ Hybridization   │ │
  │    C == C    C ── C     │ Is Aromatic     │ │
  │         │    ‖          │ Total H Count   │ │
  │         └────┘   O     │ Radical e⁻      │ │
  │              │          │ Is In Ring       │ │
  │              O ── H     │ Chirality Tag   │ │
  │                         └─────────────────┘ │
  │         Edge Features (2):                  │
  │         ┌─────────────────┐                 │
  │         │ Bond Type       │                 │
  │         │ Ring Membership │                 │
  │         └─────────────────┘                 │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
  STEP 3: PyG Data Object
  ━━━━━━━━━━━━━━━━━━━━━━━
  ┌───────────────────────────────────────┐
  │  Data(                                │
  │    x         = [N_atoms, 9]   float   │  ◀── Node feature matrix
  │    edge_index= [2, N_bonds*2] long    │  ◀── Adjacency (bidirectional)
  │    edge_attr = [N_bonds*2, 2] float   │  ◀── Edge feature matrix
  │    y         = [1]            int     │  ◀── Label (0 or 1)
  │    batch     = [N_atoms]      long    │  ◀── Graph membership
  │  )                                    │
  └───────────────────┬───────────────────┘
                      │
                      ▼
  STEP 4: GNN FORWARD PASS (see Model Architectures below)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [N_atoms, 9] ──▶ GNN Layers ──▶ Pooling ──▶ MLP ──▶ [1] logit
                      │
                      ▼
  STEP 5: PREDICTION
  ━━━━━━━━━━━━━━━━━━
  logit ──▶ σ(logit) = probability ──▶ threshold (0.5) ──▶ Active/Inactive

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## GNN Model Architectures

### Architecture Comparison

```
┌───────────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE COMPARISON                      │
├───────────────┬───────────────────┬───────────────────────────────────┤
│    GNN1       │      GNN2         │           GNN3                    │
│  (Baseline)   │  (GIN+Trans)      │     (Edge-Aware)                  │
├───────────────┼───────────────────┼───────────────────────────────────┤
│               │  ┌─────────────┐  │  ┌─────────────┐                 │
│               │  │   GINConv   │  │  │   GINConv   │                 │
│               │  │  + BatchNorm│  │  │  + BatchNorm│                 │
│               │  │  + ReLU     │  │  │  + ReLU     │                 │
│               │  └──────┬──────┘  │  └──────┬──────┘                 │
│               │         │         │         │                         │
│               │  ┌──────▼──────┐  │  ┌──────▼──────┐                 │
│               │  │ Transformer │  │  │ Transformer │                 │
│               │  │   Conv      │  │  │   Conv      │                 │
│               │  │  + BatchNorm│  │  │  + BatchNorm│                 │
│               │  │  + Reduce   │  │  │  + Reduce   │                 │
│               │  └──────┬──────┘  │  └──────┬──────┘                 │
│               │         │         │         │                         │
│  ┌─────────┐  │  ┌──────▼──────┐  │  ┌──────▼──────┐                 │
│  │GAT(3h)  │  │  │  GAT(3h)   │  │  │GAT(3h)     │                 │
│  │+BN+ReLU │  │  │  +BN+ReLU  │  │  │+edge_attr  │  ◀── uses bond  │
│  │TopK(0.8)│  │  │  TopK(0.8) │  │  │+BN+ReLU    │      features   │
│  │→ pool x1│  │  │  → pool x1 │  │  │TopK(0.8)   │                 │
│  ├─────────┤  │  ├────────────┤  │  │→ pool x1   │                 │
│  │GAT(3h)  │  │  │  GAT(3h)   │  │  ├────────────┤                 │
│  │+BN+ReLU │  │  │  +BN+ReLU  │  │  │GAT(3h)     │                 │
│  │TopK(0.5)│  │  │  TopK(0.5) │  │  │+edge_attr  │                 │
│  │→ pool x2│  │  │  → pool x2 │  │  │+BN+ReLU    │                 │
│  ├─────────┤  │  ├────────────┤  │  │TopK(0.5)   │                 │
│  │GAT(3h)  │  │  │  GAT(3h)   │  │  │→ pool x2   │                 │
│  │+BN+ReLU │  │  │  +BN+ReLU  │  │  ├────────────┤                 │
│  │TopK(0.3)│  │  │  TopK(0.3) │  │  │GAT(3h)     │                 │
│  │→ pool x3│  │  │  → pool x3 │  │  │+edge_attr  │                 │
│  └────┬────┘  │  └──────┬─────┘  │  │+BN+ReLU    │                 │
│       │       │         │        │  │TopK(0.3)   │                 │
│       ▼       │         ▼        │  │→ pool x3   │                 │
│  [x1;x2;x3]  │    [x1;x2;x3]   │  └──────┬─────┘                 │
│  concat       │    concat        │         ▼                         │
│       │       │         │        │    [x1;x2;x3]                     │
│       ▼       │         ▼        │    concat                         │
│  ┌─────────┐  │  ┌────────────┐  │         │                         │
│  │MLP(512) │  │  │ MLP(512)   │  │         ▼                         │
│  │+BN+ReLU │  │  │ +BN+ReLU   │  │  ┌────────────┐                  │
│  │Dropout  │  │  │ Dropout    │  │  │ MLP(512)   │                  │
│  │→ 1 logit│  │  │ → 1 logit  │  │  │ +BN+ReLU   │                  │
│  └─────────┘  │  └────────────┘  │  │ Dropout    │                  │
│               │                  │  │ → 1 logit  │                  │
│               │                  │  └────────────┘                  │
└───────────────┴──────────────────┴───────────────────────────────────┘
```

### GNN1 — Baseline GAT
Uses **Graph Attention Networks** with multi-head attention (3 heads) to weigh neighbor atoms. Simple but effective baseline.

### GNN2 — GIN + Transformer + GAT
Adds **GINConv** (Graph Isomorphism Network) for WL-test expressiveness — provably as powerful as the Weisfeiler-Leman graph isomorphism test. **TransformerConv** captures long-range atomic dependencies beyond local neighborhoods.

### GNN3 — Edge-Aware GIN + Transformer + GAT (Best)
Same as GNN2 but passes **edge attributes** (bond type, ring membership) through the GAT attention mechanism. Allows the model to weight neighbor contributions differently based on bond chemistry (single vs double vs aromatic).

### Key Design Decisions

```
┌─────────────────────────────────────────────────────────────────────┐
│  Multi-Scale Pooling (JK-Net Style)                                 │
│                                                                     │
│  Pool at 3 different levels → Capture local + global structure      │
│                                                                     │
│  Layer 1 (TopK 0.8) ─── x1 ──┐                                    │
│  Layer 2 (TopK 0.5) ─── x2 ──┼── CONCAT ──▶ [x1; x2; x3]         │
│  Layer 3 (TopK 0.3) ─── x3 ──┘     (preserves multi-scale info)   │
│                                                                     │
│  Each xi = GlobalMeanPool(node features at that depth)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
GNN-Based-HIV-Molecules-Classification/
│
├── model/                          # GNN architectures
│   ├── GNN1.py                     #   Baseline GAT
│   ├── GNN2.py                     #   GIN + Transformer + GAT
│   └── GNN3.py                     #   Edge-aware GIN + Transformer + GAT
│
├── data/
│   ├── raw_data/
│   │   └── HIV_data.csv            # Full dataset (41,127 molecules)
│   └── split_data/
│       ├── HIV_train.csv           # Training set
│       ├── HIV_test.csv            # Test set
│       ├── HIV_train_oversampled.csv
│       └── processed/             # Cached PyG Data objects (.pt files)
│
├── outputs/                        # Training outputs
│   ├── GNN1/
│   │   ├── best_model.pth          #   Best model checkpoint
│   │   └── confusion_matrix.png    #   Performance visualization
│   ├── GNN2/
│   └── GNN3/
│
├── app.py                          # Streamlit web application
├── main.py                         # Training pipeline (primary)
├── train.py                        # Legacy training script
├── train_optimization.py           # Optuna hyperparameter optimization
├── inference.py                    # Standalone inference script
├── dataset_featurizer.py           # SMILES → PyG graph converter
├── utils.py                        # Metrics, plotting, checkpointing
├── config.py                       # Hyperparameter search space
├── oversample_data.py              # Data oversampling utility
├── requirements.txt                # Python dependencies
└── packages.txt                    # System dependencies
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/GNN-Based-HIV-Molecules-Classification.git
cd GNN-Based-HIV-Molecules-Classification

# Install dependencies
pip install -r requirements.txt
```

### System Dependencies (for molecule visualization)

```bash
# Ubuntu/Debian
sudo apt-get install libcairo2-dev libxrender1

# macOS
brew install cairo
```

---

## Usage

### Training

```bash
# Train GNN1 (Baseline)
python3 main.py --mode train --model_type GNN1 --epochs 100

# Train GNN2 (GIN + Transformer)
python3 main.py --mode train --model_type GNN2 --epochs 100

# Train GNN3 (Edge-Aware — Best)
python3 main.py --mode train --model_type GNN3 --epochs 100

# Quick verification run (small subset)
python3 main.py --mode train --model_type GNN1 --quick_test

# Custom hyperparameters
python3 main.py --mode train --model_type GNN3 \
    --epochs 150 --batch_size 64 --lr 0.0005 \
    --patience 20 --embedding_size 512
```

### Testing

```bash
# Evaluate a trained model
python3 main.py --mode test --model_type GNN1

# Standalone inference
python3 inference.py --model_type GNN3 --weights outputs/GNN3/best_model.pth
```

### Hyperparameter Optimization

```bash
# Optuna-based optimization (30 trials)
python3 train_optimization.py --model GNN3 --n_trials 30 --epochs 20

# Or use built-in optimization mode
python3 main.py --mode optimize --model_type GNN1
```

### Streamlit Web App

```bash
streamlit run app.py
```

---

## Training Pipeline

### Training Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE (main.py)                    │
│                                                                      │
│  ┌──────────┐     ┌──────────────┐     ┌─────────────────────┐      │
│  │  Load    │────▶│  Auto-compute│────▶│  Initialize Model   │      │
│  │  Dataset │     │  pos_weight  │     │  + Adam Optimizer   │      │
│  └──────────┘     │  from labels │     │  + LR Scheduler     │      │
│                   └──────────────┘     └──────────┬──────────┘      │
│                                                    │                 │
│                   ┌────────────────────────────────┘                 │
│                   ▼                                                  │
│  ┌─────────────────────────────────────────┐                        │
│  │           TRAINING LOOP                  │                        │
│  │                                          │                        │
│  │  for epoch in range(max_epochs):         │                        │
│  │    ┌─────────────────────────────┐       │                        │
│  │    │  Train one epoch            │       │                        │
│  │    │  • Forward pass (logits)    │       │                        │
│  │    │  • BCEWithLogitsLoss        │       │                        │
│  │    │  • Backward + Adam step     │       │                        │
│  │    └──────────────┬──────────────┘       │                        │
│  │                   │                      │                        │
│  │    ┌──────────────▼──────────────┐       │                        │
│  │    │  Evaluate on test set       │       │                        │
│  │    │  • F1, Precision, Recall    │       │                        │
│  │    │  • AUC-ROC                  │       │                        │
│  │    │  • Confusion Matrix         │       │                        │
│  │    └──────────────┬──────────────┘       │                        │
│  │                   │                      │                        │
│  │    ┌──────────────▼──────────────┐       │                        │
│  │    │  ReduceLROnPlateau          │       │                        │
│  │    │  (halve LR if F1 stalls)    │       │                        │
│  │    └──────────────┬──────────────┘       │                        │
│  │                   │                      │                        │
│  │        ┌──────────▼──────────┐           │                        │
│  │        │ Best F1 improved?   │           │                        │
│  │        └──────┬────────┬─────┘           │                        │
│  │           YES │        │ NO              │                        │
│  │        ┌──────▼──────┐ │                 │                        │
│  │        │Save model   │ │ patience -= 1   │                        │
│  │        │Reset counter│ │                 │                        │
│  │        └─────────────┘ │                 │                        │
│  │                        │                 │                        │
│  │        ┌───────────────▼──────┐          │                        │
│  │        │ patience exhausted?  │          │                        │
│  │        │ → EARLY STOP         │          │                        │
│  │        └──────────────────────┘          │                        │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  OUTPUT: outputs/{model_type}/best_model.pth                         │
│          outputs/{model_type}/confusion_matrix.png                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Class Imbalance Handling

```
┌──────────────────────────────────────────────────────────────┐
│  IMBALANCE STRATEGY                                          │
│                                                              │
│  Dataset: Inactive (96.5%) ████████████████████  vs          │
│           Active   (3.5%)  █                                 │
│                                                              │
│  Solution: BCEWithLogitsLoss with pos_weight                 │
│                                                              │
│  Loss = -w_pos · y · log(σ(x)) - (1-y) · log(1 - σ(x))    │
│                                                              │
│  w_pos = n_negative / n_positive ≈ 26                        │
│  (auto-computed from training data, capped at 30)            │
│                                                              │
│  Effect: Model pays 26x MORE attention to active compounds   │
│  → Higher recall (catches more true actives)                 │
│  → Trades some precision (more false positives, acceptable)  │
└──────────────────────────────────────────────────────────────┘
```

---

## Streamlit Web App

The app provides 4 tabs:

| Tab | Description |
|-----|-------------|
| **Predict** | Single molecule prediction with molecular properties and Lipinski Ro5 check |
| **Batch Predict** | Upload CSV of SMILES → get predictions for all molecules → download results |
| **Dataset Analysis** | Real EDA with class distribution, molecular weight & LogP distributions |
| **Architecture** | Deep dive into GNN1/GNN2/GNN3 architectures and training strategy |

### App Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT APP (app.py)                    │
│                                                              │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │   Sidebar    │     │   Tab View   │     │   Backend    │  │
│  │             │     │              │     │              │  │
│  │ Model Select│────▶│ 🎯 Predict   │     │ load_model() │  │
│  │ GNN1/2/3   │     │ 📋 Batch     │────▶│ featurize()  │  │
│  │ Weights Path│     │ 📊 EDA       │     │ predict()    │  │
│  │ Arch Info   │     │ 🏗️ Blog      │     │ compute_mw() │  │
│  └─────────────┘     └──────────────┘     └──────────────┘  │
│                                                              │
│  Single Prediction Flow:                                     │
│  SMILES ──▶ RDKit Parse ──▶ featurize_smiles() ──▶ Model    │
│    │                            │                     │      │
│    ▼                            ▼                     ▼      │
│  2D Image              PyG Data Object          Probability  │
│  + Descriptors         (x, edge_index,          + Label      │
│  + Lipinski Ro5         edge_attr, batch)       + Progress   │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

### DTP AIDS Antiviral Screen

| Property | Value |
|----------|-------|
| **Source** | NIH DTP AIDS Antiviral Screen |
| **Total Molecules** | 41,127 |
| **Active (HIV inhibitors)** | 1,512 (3.5%) |
| **Inactive** | 39,615 (96.5%) |
| **Train Set** | 32,901 |
| **Test Set** | 8,238 |
| **Node Features** | 9 per atom |
| **Edge Features** | 2 per bond |

### Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| **Atomic Number** | Node | Element identity (C=6, N=7, O=8, ...) |
| **Degree** | Node | Number of bonded neighbors |
| **Formal Charge** | Node | Net electric charge on atom |
| **Hybridization** | Node | sp, sp2, sp3 orbital configuration |
| **Is Aromatic** | Node | Part of aromatic ring system |
| **H Count** | Node | Number of attached hydrogens |
| **Radical Electrons** | Node | Unpaired electrons |
| **Is In Ring** | Node | Member of any ring |
| **Chirality** | Node | 3D stereochemistry tag |
| **Bond Type** | Edge | Single (1.0), Double (2.0), Triple (3.0), Aromatic (1.5) |
| **Ring Membership** | Edge | Bond is part of a ring |

---

## Results & Metrics

### Performance Summary

| Metric | Description | Target |
|--------|-------------|--------|
| **F1 Score** | Harmonic mean of precision & recall | Primary metric |
| **Recall** | % of true actives found | Maximize (drug discovery) |
| **Precision** | % of predicted actives that are real | Balance with recall |
| **AUC-ROC** | Area under ROC curve | Overall discriminative power |

> **Note:** For drug discovery, **high recall** is more important than high precision — it's better to test a few false positives in the lab than to miss a real HIV inhibitor.

---

## Optimization Techniques

### Implemented

| Technique | How | Why |
|-----------|-----|-----|
| **BatchNorm** | After each GNN layer | Stabilizes training with aggressive TopKPooling |
| **ReLU Activations** | Between all blocks | Adds non-linearity (was missing → linear stacking) |
| **Multi-Scale Concat** | `[x1; x2; x3]` | Preserves info better than sum |
| **ReduceLROnPlateau** | Halve LR on F1 stall | Adaptive learning rate |
| **Early Stopping** | Patience-based | Prevents overfitting |
| **Auto pos_weight** | From label ratio | Data-driven imbalance handling |
| **Adam + Weight Decay** | L2 regularization | Prevents weight explosion |
| **Reduced Embedding** | 1024 → 256 | Molecules are small; avoids over-parameterization |

### Architecture Innovations

```
OPTIMIZATION: GINConv (GNN2, GNN3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standard GCN: x'_i = Σ (1/√(d_i·d_j)) · x_j · W
GINConv:      x'_i = MLP((1 + ε) · x_i + Σ x_j)

→ Provably as powerful as Weisfeiler-Leman test
→ Better structural / topology discrimination

OPTIMIZATION: TransformerConv (GNN2, GNN3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standard GNN: Only aggregates 1-hop neighbors
TransformerConv: Multi-head attention over neighbors

→ Captures long-range dependencies
→ Learns which distant atoms matter

OPTIMIZATION: Edge-Aware GAT (GNN3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standard GAT:    α_ij = softmax(a · [Wx_i || Wx_j])
Edge-Aware GAT:  α_ij = softmax(a · [Wx_i || Wx_j || We_ij])

→ Bond type influences attention weights
→ Single bonds weighted differently than aromatic bonds
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch 2.4 |
| **Graph Neural Networks** | PyTorch Geometric |
| **Chemistry** | RDKit |
| **Hyperparameter Tuning** | Optuna |
| **Web App** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Metrics** | scikit-learn |

---

## CLI Reference

```bash
python3 main.py [OPTIONS]

Options:
  --mode          {train,test,optimize}   Pipeline mode         [default: train]
  --model_type    {GNN1,GNN2,GNN3}        Architecture          [default: GNN1]
  --epochs        INT                     Max training epochs   [default: 100]
  --batch_size    INT                     Batch size            [default: 128]
  --lr            FLOAT                   Learning rate         [default: 0.001]
  --patience      INT                     Early stopping        [default: 15]
  --embedding_size INT                    Embedding dimension   [default: 256]
  --train_data    PATH                    Training CSV          [default: data/split_data/HIV_train.csv]
  --test_data     PATH                    Test CSV              [default: data/split_data/HIV_test.csv]
  --output_dir    PATH                    Output directory      [default: outputs]
  --device        {cuda,cpu}              Compute device        [default: auto]
  --quick_test                            Run on small subset
```

---

## License

This project is for educational and research purposes.

## Acknowledgments

- **Dataset**: NIH DTP AIDS Antiviral Screen
- **Framework**: PyTorch Geometric Team
- **Chemistry**: RDKit Community
