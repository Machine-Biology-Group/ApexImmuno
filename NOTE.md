# Important Notes

## Model Compatibility Issue

The pre-trained models (`Anti_inflammatory_peptide_prediction_models2.pkl`) were saved with scikit-learn version 1.0.2, but may encounter compatibility issues with newer versions (1.8.0+).

### Workaround Options:

1. **Use the original environment** with sklearn==1.0.2
   ```bash
   pip install scikit-learn==1.0.2
   ```

2. **Retrain the models** using the `global.py` script with current sklearn version

3. **Use the feature extraction pipeline** - The feature extraction code (`protein_feature`, `count_feats`) is fully functional and can be used to generate features for new models.

## What Works:

- ✓ Feature extraction (k-mers, AAindex, global descriptors)
- ✓ Data preprocessing pipeline
- ✓ Model architecture specification
- ✓ Hyperparameter tuning results (`tune_dict`)

## What Needs Attention:

- ⚠ Model pickle file compatibility
- ⚠ Requires sklearn version downgrade OR model retraining

## For Publication:

The methodology, model architecture, training data, and performance metrics are all documented and can be described in the paper. The actual prediction functionality requires either:
- Using the original environment (Python 3.9, sklearn 1.0.2)
- Retraining models with current environment

## Contact:

[Add your contact for troubleshooting]
