# Deep3D-DTA: 3D Structure-Based Drug-Target Affinity Prediction

Deep3D-DTA is an advanced deep learning framework for predicting Drug-Target Affinity (DTA). This model integrates 3D molecular structure information, 2D graph neural networks, and attention mechanisms to achieve accurate prediction of binding affinity between drugs and protein targets.


## 📋 Project Structure

```
Deep3D-DTA/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies list
├── config.py                # Configuration file
├── main_train.py            # Main training script
├── data_processing.py       # Data preprocessing module
├── model_architecture.py    # Model architecture definition
├── pdb_downloader.py       # PDB file download utility
├── graph_layers.py         # Graph neural network layers
├── resnet_encoder.py       # ResNet encoder
├── schnet_model.py         # SchNet model implementation
├── functional.py           # Functional utilities
├── utils.py               # Utility functions
├── data/                  # Data directory
│   ├── davis.csv          # Davis dataset
│   ├── kiba.csv           # KIBA dataset
│   ├── pdball.xlsx        # PDBBind data
│   └── ...
├── models/                # Model components
│   ├── edge_gat_layer.py  # Edge-aggregated GAT layer
│   ├── egret_visual.py    # EGRET visualization
│   └── pretrained_weights.dat  # Pre-trained weights
├── inputs/                # Input files directory
└── protein_sequences.txt  # Protein sequence data
```

## 🛠 Installation

### System Requirements

- Python >= 3.7
- CUDA >= 10.2 (for GPU usage)
- RAM >= 8GB

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Deep3D-DTA.git
cd Deep3D-DTA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
# PyTorch (choose based on your CUDA version)
pip install torch torchvision torch-geometric -f https://download.pytorch.org/whl/torch_stable.html

# DGL
pip install dgl dgllife

# RDKit
pip install rdkit-pypi

# Other dependencies
pip install transformers scikit-learn biopython deepchem
```

## 📊 Data Preparation

### Supported Datasets

1. **Davis**: Contains binding affinity data for 68 proteins and 442 compounds
2. **KIBA**: Contains bioactivity data for 229 targets and 2,111 drugs
3. **PDBBind**: Protein-ligand complex structure database

### Data Preprocessing

```bash
# Run data preprocessing
python data_processing.py

# Download PDB files (if needed)
python pdb_downloader.py
```

## 🏃‍♂️ Usage

### Training

```bash
# Train with default configuration
python main_train.py

```

