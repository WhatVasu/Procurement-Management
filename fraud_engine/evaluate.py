"""Evaluate the persisted Fraud Engine on held-out external cases."""

from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .fraud_engine import FraudEngine


def main(data_dir="fraud_engine/data", artifact_dir="fraud_engine/artifacts"):
    root = Path(data_dir)
    engine = FraudEngine.load(Path(artifact_dir) / "fraud_model.joblib")
    predictions = engine.predict_frame(pd.read_csv(root / "fraud_test_cases.csv"))
    truth = pd.read_csv(root / "fraud_ground_truth.csv")
    result = predictions.merge(truth, on="po_id", validate="one_to_one")
    print(json.dumps({
        "rows": len(result),
        "accuracy": round(float(accuracy_score(result.expected_fraud_label, result.fraud_flag)), 4),
        "precision": round(float(precision_score(result.expected_fraud_label, result.fraud_flag, zero_division=0)), 4),
        "recall": round(float(recall_score(result.expected_fraud_label, result.fraud_flag, zero_division=0)), 4),
        "f1": round(float(f1_score(result.expected_fraud_label, result.fraud_flag, zero_division=0)), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
