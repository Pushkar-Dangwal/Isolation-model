#!/usr/bin/env python3
"""Flask app for interacting with the full-dataset fraud model."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, request

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fraud_detector import FraudDetector


MODEL_NAME = "production_fraud_detector_20260402_143243"
MODEL_VERSION = "20260402_143243"

app = Flask(__name__)
detector: FraudDetector | None = None


def _load_model() -> FraudDetector:
    loaded_detector = FraudDetector()
    model_root = str(Path("models") / MODEL_NAME)
    loaded_detector.load_model(
        filepath=model_root,
        version=MODEL_VERSION,
        verify_integrity=False,
        load_pipeline=False,
    )
    return loaded_detector


def _normalize_transaction(tx: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    normalized = dict(tx)

    if not normalized.get("transaction_id"):
        normalized["transaction_id"] = f"tx_{int(datetime.now().timestamp())}_{index}"

    if not normalized.get("timestamp"):
        normalized["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    required_defaults = {
        "sender_account": "sender_unknown",
        "receiver_account": "receiver_unknown",
        "transaction_type": "payment",
        "merchant_category": "retail",
        "location": "unknown",
        "device_used": "mobile",
    }

    for key, default_value in required_defaults.items():
        if not normalized.get(key):
            normalized[key] = default_value

    if "amount" not in normalized or normalized["amount"] in (None, ""):
        normalized["amount"] = 0.0

    normalized["amount"] = float(normalized["amount"])
    return normalized


def _predict_transactions(transactions: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if detector is None:
        raise RuntimeError("Model is not loaded")

    normalized = [_normalize_transaction(tx, idx) for idx, tx in enumerate(transactions)]
    input_df = pd.DataFrame(normalized)

    predictions = detector.predict(
        input_df,
        transaction_id_column="transaction_id",
        return_probabilities=True,
        return_risk_levels=True,
        return_explanations=True,
    )

    output_columns = [
        "transaction_id",
        "fraud_probability",
        "anomaly_score",
        "risk_level",
        "fraud_prediction",
        "explanation",
        "processed_at",
        "model_version",
        "has_error",
    ]

    existing_columns = [col for col in output_columns if col in predictions.columns]
    results = predictions[existing_columns].copy()
    results["processed_at"] = results["processed_at"].astype(str)

    return results.to_dict(orient="records")


@app.route("/", methods=["GET"])
def api_info():
    return jsonify(
        {
            "service": "fraud-detector-api",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "endpoints": {
                "health": "GET /health",
                "predict": "POST /predict",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    if detector is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "model_name": MODEL_NAME, "model_version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    if isinstance(payload, dict) and "transactions" in payload:
        transactions = payload["transactions"]
    elif isinstance(payload, list):
        transactions = payload
    elif isinstance(payload, dict):
        transactions = [payload]
    else:
        return jsonify({"error": "Invalid JSON payload format"}), 400

    if not transactions:
        return jsonify({"error": "No transactions provided"}), 400

    try:
        results = _predict_transactions(transactions)
        return jsonify(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "count": len(results),
                "predictions": results,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    detector = _load_model()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
else:
    detector = _load_model()
