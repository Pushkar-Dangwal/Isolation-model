# Financial Fraud Detection System

A comprehensive machine learning solution for identifying fraudulent transactions in real-time while minimizing false positives.

## Project Structure

```
fraud-detection-system/
├── src/                    # Source code
│   ├── __init__.py
│   └── config.py          # Configuration and logging setup
├── tests/                 # Test files
│   └── __init__.py
├── data/                  # Training data and datasets
├── models/                # Trained model artifacts
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Features

- **Data Preprocessing**: Handles raw transaction data cleaning and transformation
- **Feature Engineering**: Creates sophisticated fraud-indicative features
- **Anomaly Detection**: Uses Deep Isolation Forest for unsupervised anomaly scoring
- **Supervised Classification**: Employs gradient boosting for final fraud prediction
- **Risk Scoring**: Provides interpretable risk levels and threshold optimization

## Usage

The system is designed to process transactions through a multi-layered pipeline:

1. Data preprocessing and cleaning
2. Advanced feature engineering
3. Anomaly detection scoring
4. Supervised classification
5. Risk level assignment

## Configuration

System configuration is managed through `src/config.py`, including:
- Logging setup
- Model hyperparameters
- Feature engineering parameters
- Risk scoring thresholds

## Testing

The project uses both unit tests and property-based tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Development

This project follows the spec-driven development methodology with comprehensive requirements, design documentation, and implementation tasks defined in `.kiro/specs/fraud-detection-system/`.