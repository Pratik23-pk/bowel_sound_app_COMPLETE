# Models Directory

This directory should contain your trained model files and normalization parameters.

## Required Files

Place the following files in this directory:

1. **crnn_best.keras** - Your trained CRNN model file
2. **standardization_mean.npy** - Mean values for standardization
3. **standardization_std.npy** - Standard deviation values for standardization
4. **optimal_threshold.npy** (optional) - Optimal classification threshold

## File Descriptions

### crnn_best.keras
The trained Keras model file containing the CRNN architecture for bowel sound classification.

### standardization_mean.npy
NumPy array containing the mean values computed from your training data, used for standardizing input features.

### standardization_std.npy
NumPy array containing the standard deviation values computed from your training data, used for standardizing input features.

### optimal_threshold.npy (optional)
The optimal threshold value for binary classification. If not provided, the default threshold of 0.5 will be used.

## Notes

- All files should be generated from your training pipeline
- Do not modify these files manually
- Keep backups of your model files in a safe location
