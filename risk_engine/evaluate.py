"""Evaluate persisted predictions against the evaluation-only future labels."""

from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from risk_engine.riskwise_engine import FeatureBuilder, RiskPredictor, RiskWiseData


def main(data_dir: str = ".", artifact_dir: str = "artifacts") -> None:
    root = Path(data_dir)
    data = RiskWiseData.load(root)
    cases = pd.read_csv(root / "new_po_test_cases.csv")
    truth = pd.read_csv(root / "new_po_ground_truth.csv")
    artifact = Path(artifact_dir)
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    predictor = RiskPredictor.load(artifact / "riskwise_model.joblib")
    features = FeatureBuilder(data).build(cases, cases)
    predictions = predictor.predict_frame(features)
    result = predictions.merge(truth, on="po_id", validate="one_to_one")
    print(json.dumps({
        "rows": len(result),
        "accuracy_level": round(float(accuracy_score(result.ground_truth_risk_level, result.risk_level)), 4),
        "f1_weighted_level": round(float(f1_score(result.ground_truth_risk_level, result.risk_level, average="weighted")), 4),
        "high_precision": round(float(precision_score(result.ground_truth_risk_level == "HIGH", result.risk_level == "HIGH", zero_division=0)), 4),
        "high_recall": round(float(recall_score(result.ground_truth_risk_level == "HIGH", result.risk_level == "HIGH", zero_division=0)), 4),
        "score_mae": round(float((result.risk_score - result.ground_truth_risk_score).abs().mean()), 4),
    }, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="shared_data")
    parser.add_argument("--artifact-dir", default="risk_engine/artifacts")
    args = parser.parse_args()
    main(args.data_dir, args.artifact_dir)
