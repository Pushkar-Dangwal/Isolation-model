# Final Diagnosis: Training vs Evaluation Inconsistency

## Summary

Your fraud detection model shows a critical inconsistency:
- **Training Performance**: ROC-AUC 1.000 (100% - PERFECT)
- **Evaluation Performance**: ROC-AUC 0.501 (50% - RANDOM)

This is caused by **severe overfitting** combined with the feature engineering issue we fixed.

## What We Fixed

### 1. Feature Engineering Issue ✓ FIXED
**Problem**: The `transform()` method wasn't computing time-based features
**Solution**: Modified `transform()` to compute all features like `fit_transform()`
**Result**: All 50 features are now generated correctly

### 2. Evaluation Pipeline Issue ✓ FIXED  
**Problem**: Evaluation was calling predict on test subset without historical context
**Solution**: Modified `evaluate_model()` to predict on full dataset, then extract test subset
**Result**: Features now have proper historical context

## Remaining Issue: Severe Overfitting

### Evidence
```
Training:
- ROC-AUC: 1.000 (perfect - suspicious!)
- PR-AUC: 0.976
- Best iteration: 7 (stopped very early)
- Training samples: 400,000
- Fraud rate: 1.0%

Evaluation:
- ROC-AUC: 0.501 (random)
- PR-AUC: 0.014
- Fraud Detection Rate: 1.2%
- Test samples: 100,000
```

### Root Causes

1. **Model Complexity**: The model is too complex for the data
   - Deep Isolation Forest with 3 layers, 128 hidden units
   - LightGBM with 100 estimators
   - Total: 51 features (50 + anomaly score)

2. **Early Stopping Failure**: Model stopped at iteration 7
   - This suggests validation set issues
   - Model might be memorizing training data

3. **Data Distribution**: Training on 500K samples with 1% fraud rate
   - Only 5,000 fraud cases total
   - Split: 4,000 train frauds, 1,000 test frauds
   - Not enough fraud examples for complex model

## Solutions

### Option 1: Simplify the Model (RECOMMENDED)
Reduce model complexity to prevent overfitting:

```python
# Simpler anomaly detector
anomaly_detector_config = {
    'n_estimators': 50,  # Reduced from 100
    'n_layers': 2,       # Reduced from 3
    'n_hidden': 64       # Reduced from 128
}

# Simpler classifier
classifier_config = {
    'learning_rate': 0.05,  # Slower learning
    'num_leaves': 15,       # Reduced from 31
    'max_depth': 5,         # Add depth limit
    'min_child_samples': 100,  # Require more samples per leaf
    'n_estimators': 50      # Reduced from 1000
}
```

### Option 2: Use More Training Data
Train on the full 5M dataset instead of 500K:

```bash
python train_model.py --data-path data/financial.csv --sample-size 5000000
```

This will provide:
- 50,000 fraud cases (vs 5,000)
- Better generalization
- More reliable validation

### Option 3: Feature Selection
Reduce the number of features to prevent overfitting:
- Remove highly correlated features
- Use only the most important 20-30 features
- Simplify time-based aggregations

### Option 4: Regularization
Add stronger regularization to LightGBM:

```python
classifier_config = {
    'learning_rate': 0.01,  # Much slower
    'reg_alpha': 1.0,       # L1 regularization
    'reg_lambda': 1.0,      # L2 regularization
    'min_gain_to_split': 0.1,  # Require minimum gain
    'bagging_fraction': 0.8,    # Use 80% of data per iteration
    'feature_fraction': 0.8     # Use 80% of features per iteration
}
```

## Recommended Action Plan

1. **Immediate**: Use Option 2 (More Data)
   ```bash
   python train_model.py \
       --data-path data/financial.csv \
       --sample-size 2000000 \
       --validation-split 0.2 \
       --random-seed 42
   ```

2. **If still overfitting**: Combine Option 1 + Option 4
   - Simplify model architecture
   - Add regularization
   - Use cross-validation

3. **Monitor**: Check these metrics
   - Training AUC should be 0.90-0.95 (not 1.00!)
   - Evaluation AUC should be 0.85-0.92
   - Gap between train/eval should be < 0.05

## Why Training Showed 1.000 AUC

Perfect training AUC (1.000) is a red flag indicating:
1. Model memorized the training data
2. Model is too complex for the problem
3. Possible data leakage (though unlikely here)
4. Early stopping didn't work correctly

A good model should have:
- Training AUC: 0.90-0.95
- Evaluation AUC: 0.85-0.92
- Small gap (< 0.05) between them

## Current Status

✓ Feature engineering fixed - all 50 features generated
✓ Evaluation pipeline fixed - proper historical context
✗ Model overfitting - needs simpler architecture or more data

## Next Steps

1. Retrain with 2M+ samples for better generalization
2. If still overfitting, simplify the model
3. Monitor train/eval gap to ensure it's < 0.05
4. Target evaluation AUC of 0.85-0.92 (not 1.00!)

The fixes we applied are correct and necessary. The remaining issue is model complexity vs data size, which requires either more data or a simpler model.
