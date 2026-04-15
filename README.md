# Anti-inflammatory Peptide Predictor

This package contains the trained models and scripts for predicting anti-inflammatory activity of peptide sequences.

## Overview

The predictor uses an ensemble of 10 Extra Trees Classifier models trained on anti-inflammatory peptide data from AIPpred database (http://www.thegleelab.org/AIPpred/).

### Model Architecture
- **Model Type**: Extra Trees Classifier (tree-based ensemble method)
- **Number of Models**: 10 (ensemble)
- **Feature Extraction**:
  - Amino acid composition (1-mer and 2-mer)
  - Gap k-mers (up to varying gap lengths per model)
  - AAindex descriptors (566 physicochemical properties)
  - Global descriptors (length, charge, isoelectric point, aromaticity, aliphatic index, hydrophobic ratio)

### Training Data
- **Source**: AIPpred database (http://www.thegleelab.org/AIPpred/)
- **Positive samples**: Anti-inflammatory peptides (benchmarking and independent datasets)
- **Negative samples**: Non-anti-inflammatory peptides

### Performance
Cross-validation performance (5-fold stratified CV, 5 repeats):
- **AUROC**: ~0.80
- **AUPR**: ~0.72
- **Accuracy**: ~0.74

Top model configurations:
| Gap | n_estimators | max_depth | AUC | AUPR | ACC |
|-----|--------------|-----------|-----|------|-----|
| 4   | 4098         | 128       | 0.800 | 0.720 | 0.743 |
| 4   | 8192         | 64        | 0.796 | 0.719 | 0.741 |
| 10  | 2048         | 64        | 0.796 | 0.718 | 0.739 |

## Files Included

- `predict_anti_inflammatory.py` - Main prediction script
- `global.py` - Training script with full pipeline
- `utils.py` - Utility functions for feature extraction
- `FC.py` - Fully connected model (for deep learning alternative)
- `aaindex1.csv` - AAindex database file
- `Anti_inflammatory_peptide_prediction_models2.pkl` - Trained ensemble models (scalers + models)
- `tune_dict` - Hyperparameter tuning results
- `test_predictor.py` - Test script
- `README.md` - This file

## Requirements

```bash
pip install numpy pandas scikit-learn modlamp biopython
```

Python packages:
- numpy
- pandas
- scikit-learn>=1.0.2
- modlamp
- biopython

Pretrained model: https://zenodo.org/records/19596074

## Usage

### Quick Start

```python
from predict_anti_inflammatory import predict_anti_inflammatory

# Example peptide sequences
peptides = [
    "SIIKSLAALLTKLAIIVK",
    "KYLLNYTAKLIIKKLAKI",
    "PKLIYKIGALIKAVKKI"
]

# Get predictions
probabilities = predict_anti_inflammatory(peptides)

# Print results
for seq, prob in zip(peptides, probabilities):
    print(f"{seq}: {prob:.4f}")
```

### Command Line Usage

```bash
python predict_anti_inflammatory.py
```

This will run predictions on example peptides included in the script.

### Testing

Run the test script to verify installation:

```bash
python test_predictor.py
```

## Output

The predictor returns probability scores between 0 and 1, where:
- **Higher values (closer to 1)**: Stronger predicted anti-inflammatory activity
- **Lower values (closer to 0)**: Weaker predicted anti-inflammatory activity

Typical threshold: **≥0.85** for high-confidence predictions (based on filtering used in the generation pipeline)

## Model Details

### Feature Engineering
The model uses a comprehensive set of features:
1. **k-mer composition**: 1-mer (20 features) and 2-mer (400 features)
2. **Gap k-mers**: Variable gap lengths (1 to 10, depending on model configuration)
3. **AAindex properties**: 566 physicochemical and biochemical properties
4. **Global descriptors**: 6 sequence-level features

Total feature dimensions: ~3,000-7,000 (varies by model based on gap parameter)

### Training Details
- **Algorithm**: Extra Trees Classifier (sklearn.ensemble.ExtraTreesClassifier)
- **Ensemble size**: 10 models with different configurations
- **Sampling strategy**: Balanced sampling (oversampling positive class during training)
- **Standardization**: StandardScaler applied to features
- **Cross-validation**: 5-fold stratified, 5 repeats

### Hyperparameter Tuning
Grid search was performed over:
- Gap k-mer parameter: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
- Number of trees: {1024, 2048, 4098, 8192, 16384}
- Maximum depth: {32, 64, 128}

Total configurations evaluated: 150


