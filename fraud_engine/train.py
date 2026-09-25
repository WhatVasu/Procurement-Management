"""Compare and persist Fraud Detection Engine models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .audit import audit
from .fraud_engine import FraudEngine, FraudThresholds


def main(data_dir: str = "fraud_engine/data", artifact_dir: str = "fraud_engine/artifacts") -> None:
    root, out = Path(data_dir), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit(root)
    transactions = pd.read_csv(root / "fraud_transactions.csv")
    labels = pd.read_csv(root / "fraud_labels.csv")
    frame = transactions.merge(labels[["po_id", "fraud_label"]], on="po_id", validate="one_to_one")
    feature_names = [c for c in transactions.columns if c not in {"po_id", "supplier_id", "transaction_date"}]
    x, y = frame[feature_names], frame.fraud_label
    categorical = [c for c in feature_names if not is_numeric_dtype(x[c])]
    numeric = [c for c in feature_names if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    preprocessor.fit(x_train)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(preprocessor.transform(x_train), y_train)
        pred = model.predict(preprocessor.transform(x_test))
        proba = model.predict_proba(preprocessor.transform(x_test))[:, list(model.classes_).index(1)]
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "fraud_recall": round(float(recall_score(y_test, pred, pos_label=1)), 4),
        }
    selected = max(metrics, key=lambda name: metrics[name]["f1"])
    model = models[selected]
    model.fit(preprocessor.transform(x), y)
    artifact = {"model": model, "preprocessor": preprocessor, "feature_names": feature_names,
                "thresholds": FraudThresholds().__dict__, "model_name": selected}
    joblib.dump(artifact, out / "fraud_model.joblib")
    engine = FraudEngine(artifact)
    cases = pd.read_csv(root / "fraud_test_cases.csv")
    truth = pd.read_csv(root / "fraud_ground_truth.csv")
    predictions = engine.predict_frame(cases)
    evaluated = predictions.merge(truth, on="po_id", validate="one_to_one")
    results = {
        "training": {"rows": len(frame), "stratified_test_rows": len(y_test), "models": metrics, "selected_model": selected,
                     "excluded_from_training": ["fraud_test_cases.csv", "fraud_ground_truth.csv"]},
        "thresholds": FraudThresholds().__dict__,
        "external_evaluation": {
            "rows": len(evaluated),
            "accuracy": round(float(accuracy_score(evaluated.expected_fraud_label, evaluated.fraud_flag)), 4),
            "precision": round(float(precision_score(evaluated.expected_fraud_label, evaluated.fraud_flag, zero_division=0)), 4),
            "recall": round(float(recall_score(evaluated.expected_fraud_label, evaluated.fraud_flag, zero_division=0)), 4),
            "f1": round(float(f1_score(evaluated.expected_fraud_label, evaluated.fraud_flag, zero_division=0)), 4),
        },
    }
    predictions.to_csv(out / "fraud_predictions.csv", index=False)
    (out / "fraud_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="fraud_engine/data")
    parser.add_argument("--artifact-dir", default="fraud_engine/artifacts")
    args = parser.parse_args()
    main(args.data_dir, args.artifact_dir)
