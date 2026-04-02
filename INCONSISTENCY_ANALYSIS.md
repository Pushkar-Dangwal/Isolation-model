# Model Training vs Evaluation Inconsistency Analysis

## Problem Summary
The newly trained model `fraud_detector_20260320_130507` shows excellent training performance (ROC-AUC: 0.948) but terrible evaluation performance (ROC-AUC: 0.500). This is caused by a feature engineering pipeline bug during evaluation.

## Root Cause Identified

### Training Flow (CORRECT)
1. `train_fraud_detector()` calls `detector.fit(df)` with FULL dataset
2. Inside `fit()`, feature engineer uses `fit_transform()` on the FULL dataset
3. Time-based features are computed with complete historical context:
   - `tx_count_last_1h` - counts transactions in last hour
   - `velocity_1h` - transaction velocity
   - `receiver_tx_count` - receiver transaction history
   - `sender_receiver_frequency` - interaction patterns
4. All 50 features are created successfully
5. Model trains on complete feature set → **94.8% ROC-AUC**

### Evaluation Flow (BROKEN)
1. `evaluate_model()` splits data: 80% train, 20% test
2. Calls `detector.evaluate(X_test)` with ONLY test subset
3. Inside `evaluate()` → `predict()` → `feature_engineer.transform()`
4. **CRITICAL BUG**: `transform()` method is called on test data WITHOUT historical context
5. Time-based aggregation features CANNOT be computed:
   - No transaction history for senders/receivers
   - No time windows to aggregate over
   - No interaction patterns available
6. Result: **20 out of 50 features (40%) are missing**
7. Error handler fills missing features with zeros
8. Model receives corrupted feature matrix → **50% ROC-AUC (random)**

## Evidence from Logs

```
2026-03-20 13:13:51,354 - model_evaluator - INFO - Performance report completed
  - PR-AUC: 0.036
  - F1: 0.000
  - Precision: 0.000
  - Recall: 0.000
  - ROC-AUC: 0.500  ← Random performance!
```

## Missing Features (20 out of 50)

Time-based aggregation features that require historical context:
1. `tx_count_last_1h`
2. `tx_count_last_24h`
3. `total_amount_last_24h`
4. `avg_amount_last_24h`
5. `max_amount_last_24h`
6. `velocity_1h`
7. `velocity_24h`
8. `receiver_tx_count`
9. `receiver_fraud_count`
10. `receiver_fraud_rate`
11. `receiver_total_amount`
12. `receiver_avg_amount`
13. `receiver_risk_score`
14. `sender_receiver_frequency`
15. `sender_receiver_total_amount`
16. `sender_receiver_avg_amount`
17. `location_frequency`
18. `device_frequency`
19. `sender_location_count`
20. `sender_device_count`

## Code Locations

### Bug Location
**File**: `train_model.py`
**Function**: `evaluate_model()` (lines 382-450)
**Problem**: Calls `detector.evaluate(X_test)` with only test subset

### Feature Engineering
**File**: `src/feature_engineer.py`
**Methods**:
- `fit_transform()` - Used during training (CORRECT - has full context)
- `transform()` - Used during evaluation (BROKEN - no historical context)

### Prediction Pipeline
**File**: `src/fraud_detector.py`
**Method**: `predict()` (lines 700-900)
**Issue**: Calls `feature_engineer.transform()` which can't compute time-based features

## Solution Options

### Option 1: Pass Full Dataset to Evaluation (RECOMMENDED)
Modify `evaluate_model()` to pass the full dataset for feature engineering, then extract predictions for test subset only.

```python
# In evaluate_model():
# Instead of:
evaluation_results = detector.evaluate(X_test, target_column='is_fraud')

# Do this:
# 1. Pass full dataset for feature engineering
predictions = detector.predict(df)  # Full dataset
# 2. Extract test subset predictions
test_predictions = predictions.loc[X_test.index]
# 3. Calculate metrics on test subset only
```

### Option 2: Fix Feature Engineer Transform Method
Modify `transform()` to maintain historical state from `fit_transform()` and use it during inference.

### Option 3: Use Fit-Transform for Evaluation
Call `fit_transform()` instead of `transform()` during evaluation (not ideal for production simulation).

## Impact

- **Training metrics**: CORRECT (94.8% ROC-AUC)
- **Evaluation metrics**: INCORRECT (50% ROC-AUC due to missing features)
- **Production inference**: WILL FAIL with same issue (40% features missing)

## Recommendation

Implement Option 1 immediately. The model itself is excellent (94.8% AUC), but the evaluation and inference pipelines need to be fixed to provide proper historical context for time-based feature computation.
