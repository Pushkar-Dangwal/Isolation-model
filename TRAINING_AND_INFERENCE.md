# Fraud Detection System - Training and Inference Guide

This guide explains how to use the training and inference scripts for the fraud detection system.

## Overview

The fraud detection system provides two main scripts:

1. **`train_model.py`** - End-to-end training pipeline for the fraud detection model
2. **`inference.py`** - Real-time and batch inference pipeline for scoring transactions

## Prerequisites

Ensure you have:
- Python 3.8+ installed
- All required dependencies (see `requirements.txt`)
- Training data in the expected format
- Sufficient system resources (RAM and CPU)

## Training Script (`train_model.py`)

### Purpose
Trains a complete fraud detection pipeline including data preprocessing, feature engineering, anomaly detection, supervised classification, and risk scoring.

### Basic Usage

```bash
# Train with default settings
python train_model.py --data-path data/financial.csv

# Train with custom configuration
python train_model.py \
  --data-path data/financial.csv \
  --model-name my_fraud_model \
  --enable-reproducibility \
  --optimize-thresholds \
  --save-evaluation \
  --verbose
```

### Key Arguments

#### Data Arguments
- `--data-path`: Path to training CSV file (required)
- `--sample-size`: Number of samples to use (optional, uses all if not specified)
- `--validation-split`: Fraction for validation (default: 0.2)

#### Model Arguments
- `--model-name`: Name for the trained model (auto-generated if not provided)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--enable-reproducibility`: Enable strict reproducibility controls
- `--strict-determinism`: Enable fully deterministic mode (slower but reproducible)

#### Training Arguments
- `--optimize-thresholds`: Perform threshold optimization (default: True)
- `--enable-error-handling`: Enable comprehensive error handling (default: True)
- `--n-jobs`: Number of parallel jobs (default: -1 for all cores)

#### Output Arguments
- `--output-dir`: Directory to save model (default: models/)
- `--save-evaluation`: Save detailed evaluation results (default: True)
- `--save-plots`: Save evaluation plots and visualizations

### Expected Data Format

The training data CSV should contain these columns:

```csv
transaction_id,timestamp,sender_account,receiver_account,amount,transaction_type,merchant_category,location,device_used,is_fraud
tx_000001,2024-01-01 10:30:00,sender_0001,receiver_0001,150.00,transfer,grocery,NYC,mobile,0
tx_000002,2024-01-01 11:45:00,sender_0002,receiver_0002,2500.00,payment,online,LA,web,1
...
```

### Training Process

The training script performs these steps:

1. **Data Loading and Validation**
   - Loads CSV data with optimized dtypes
   - Validates required columns and data quality
   - Reports dataset statistics

2. **Model Training**
   - Initializes fraud detection pipeline
   - Trains all components (preprocessing, feature engineering, anomaly detection, classification)
   - Optimizes risk thresholds

3. **Model Evaluation**
   - Evaluates on validation set
   - Generates comprehensive metrics
   - Creates performance reports

4. **Model Saving**
   - Saves trained model with metadata
   - Saves evaluation results
   - Saves training configuration

### Output Files

After training, you'll find:
- `models/{model_name}.joblib` - Trained model
- `models/{model_name}_evaluation.json` - Evaluation results
- `models/{model_name}_config.json` - Training configuration
- `models/{model_name}_metadata.json` - Comprehensive metadata

## Inference Script (`inference.py`)

### Purpose
Scores new transactions using a trained fraud detection model, supporting both real-time and batch processing.

### Basic Usage

```bash
# Batch inference from CSV
python inference.py \
  --model-path models/my_fraud_model \
  --input-data new_transactions.csv \
  --output-file predictions.csv

# Real-time inference with benchmarking
python inference.py \
  --model-path models/my_fraud_model \
  --input-data new_transactions.csv \
  --real-time \
  --benchmark \
  --enable-explanations

# Single transaction inference
python inference.py \
  --model-path models/my_fraud_model \
  --single-transaction '{"transaction_id":"tx_001","timestamp":"2024-01-15 14:30:00","sender_account":"sender_001","receiver_account":"receiver_001","amount":1500.00,"transaction_type":"transfer","merchant_category":"online","location":"NYC","device_used":"mobile"}' \
  --enable-explanations
```

### Key Arguments

#### Model Arguments
- `--model-path`: Path to trained model (required)
- `--model-version`: Specific model version (optional, loads latest)

#### Input Arguments
- `--input-data`: Path to input CSV file
- `--input-json`: Path to input JSON file
- `--single-transaction`: JSON string with single transaction

#### Processing Arguments
- `--batch-size`: Batch size for large datasets (default: 1000)
- `--real-time`: Enable real-time processing mode
- `--enable-explanations`: Include prediction explanations

#### Output Arguments
- `--output-file`: CSV output path
- `--output-json`: JSON output path
- `--output-format`: Output format (csv/json/both)

#### Performance Arguments
- `--benchmark`: Run performance benchmarking
- `--profile-memory`: Profile memory usage

### Input Data Format

For inference, the data should contain the same columns as training (except `is_fraud`):

```csv
transaction_id,timestamp,sender_account,receiver_account,amount,transaction_type,merchant_category,location,device_used
tx_new_001,2024-01-15 14:30:00,sender_0001,receiver_0001,150.00,transfer,grocery,NYC,mobile
tx_new_002,2024-01-15 15:45:00,sender_0002,receiver_0002,2500.00,payment,online,LA,web
...
```

### Processing Modes

#### Batch Processing (Default)
- Processes transactions in batches
- Optimized for throughput
- Suitable for large datasets
- Lower per-transaction latency

#### Real-time Processing
- Processes transactions individually
- Optimized for latency
- Provides detailed timing metrics
- Suitable for production APIs

### Output Format

The inference results include:

```csv
transaction_id,fraud_probability,anomaly_score,risk_level,fraud_prediction,processed_at,model_version
tx_new_001,0.05,0.1,low,0,2024-01-15T14:30:00,fraud_detector_v1.0_20240115
tx_new_002,0.85,0.9,high,1,2024-01-15T15:45:00,fraud_detector_v1.0_20240115
```

With explanations enabled:
```csv
...,explanation
...,Low fraud risk (probability: 0.05) - transaction appears legitimate
...,High fraud risk (probability: 0.85) due to unusual transaction patterns (anomaly score: 0.90)
```

## Performance Considerations

### Training Performance
- **Memory Usage**: ~400MB for 5M transactions
- **Training Time**: 10-30 minutes depending on hardware
- **CPU Usage**: Utilizes all available cores by default
- **Reproducibility**: Adds ~10-20% overhead when enabled

### Inference Performance
- **Batch Mode**: 1000-5000 transactions/second
- **Real-time Mode**: 100-500 transactions/second
- **Memory Usage**: ~100-200MB for loaded model
- **Latency**: 1-10ms per transaction (real-time mode)

### Optimization Tips

1. **For Training**:
   - Use `--sample-size` for development/testing
   - Disable `--strict-determinism` for faster training
   - Use SSD storage for data files
   - Ensure sufficient RAM (8GB+ recommended)

2. **For Inference**:
   - Use batch mode for high throughput
   - Use real-time mode for low latency
   - Adjust `--batch-size` based on memory constraints
   - Pre-load model to avoid startup costs

## Error Handling

Both scripts include comprehensive error handling:

### Common Issues and Solutions

1. **Memory Errors**:
   - Reduce batch size or sample size
   - Use more efficient data types
   - Process data in chunks

2. **Model Loading Errors**:
   - Verify model path and version
   - Check model integrity
   - Ensure compatible Python/library versions

3. **Data Format Errors**:
   - Validate column names and types
   - Check for missing required columns
   - Handle missing values appropriately

4. **Performance Issues**:
   - Monitor system resources
   - Adjust parallelization settings
   - Use appropriate processing mode

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Generate sample data (for testing)
python example_usage.py

# 2. Train the model
python train_model.py \
  --data-path training_data.csv \
  --model-name production_fraud_model \
  --enable-reproducibility \
  --optimize-thresholds \
  --save-evaluation \
  --verbose

# 3. Run batch inference
python inference.py \
  --model-path models/production_fraud_model \
  --input-data inference_data_clean.csv \
  --output-file predictions.csv \
  --output-format both \
  --enable-explanations \
  --benchmark

# 4. Check results
head predictions.csv
```

## Integration with Production Systems

### API Integration
The inference script can be integrated into production APIs:

```python
from fraud_detector import FraudDetector

# Load model once at startup
detector = FraudDetector()
detector.load_model("models/production_fraud_model")

# Score transactions in API endpoint
def score_transaction(transaction_data):
    result = detector.predict_single(transaction_data)
    return {
        'fraud_probability': result['fraud_probability'],
        'risk_level': result['risk_level'],
        'recommendation': get_recommendation(result['risk_level'])
    }
```

### Batch Processing
For batch processing systems:

```python
# Process daily batch
python inference.py \
  --model-path models/production_fraud_model \
  --input-data daily_transactions.csv \
  --output-file daily_predictions.csv \
  --batch-size 5000
```

### Monitoring and Maintenance

1. **Model Performance**: Monitor precision, recall, and false positive rates
2. **Data Drift**: Check for changes in transaction patterns
3. **System Health**: Monitor memory usage and processing times
4. **Model Updates**: Retrain periodically with new data

## Troubleshooting

### Common Commands for Debugging

```bash
# Check model information
python -c "
from src.fraud_detector import FraudDetector
detector = FraudDetector()
detector.load_model('models/my_model')
print(detector.get_model_info())
"

# Validate input data
python inference.py \
  --model-path models/my_model \
  --input-data test_data.csv \
  --verbose \
  --log-level DEBUG

# Run with error handling disabled (for debugging)
python train_model.py \
  --data-path data.csv \
  --model-name debug_model \
  --log-level DEBUG
```

### Getting Help

```bash
# Training script help
python train_model.py --help

# Inference script help
python inference.py --help

# Example usage
python example_usage.py
```

## Advanced Usage

### Custom Configuration
Both scripts support extensive configuration through command-line arguments. For production deployments, consider creating wrapper scripts with your specific configurations.

### Reproducibility
Enable reproducibility for research and compliance:

```bash
python train_model.py \
  --data-path data.csv \
  --enable-reproducibility \
  --strict-determinism \
  --random-seed 12345
```

### Performance Benchmarking
Use benchmarking to optimize for your hardware:

```bash
python inference.py \
  --model-path models/my_model \
  --input-data test_data.csv \
  --benchmark \
  --profile-memory
```

This will generate detailed performance reports to help optimize your deployment.