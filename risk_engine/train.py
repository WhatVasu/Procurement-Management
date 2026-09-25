"""Train, compare, evaluate, and persist RiskWise models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype

from risk_engine.audit import audit
from risk_engine.riskwise_engine import FeatureBuilder, RiskPredictor, RiskWiseData, model_columns


def metrics(model, preprocessor, x, y) -> dict:
    pred = model.predict(preprocessor.transform(x))
    proba = model.predict_proba(preprocessor.transform(x))
    classes = list(model.classes_)
    high = classes.index("HIGH")
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision_weighted": round(float(precision_score(y, pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y, pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "roc_auc_ovr_weighted": round(float(roc_auc_score(pd.get_dummies(y).reindex(columns=classes, fill_value=0), proba, multi_class="ovr", average="weighted")), 4),
        "high_precision": round(float(precision_score(y == "HIGH", pred == "HIGH", zero_division=0)), 4),
        "high_recall": round(float(recall_score(y == "HIGH", pred == "HIGH", zero_division=0)), 4),
        "high_predicted_probability_mean": round(float(proba[:, high].mean()), 4),
    }


def main(data_dir: str = ".", artifact_dir: str = "artifacts") -> None:
    root = Path(data_dir)
    out = Path(artifact_dir)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(exist_ok=True)
    audit(root)
    data = RiskWiseData.load(root)
    builder = FeatureBuilder(data)
    features = builder.build(data.purchase_orders)
    labels = pd.read_csv(root / "risk_labels.csv")
    frame = features.merge(labels[["po_id", "risk_score", "risk_level"]], on="po_id", validate="one_to_one")
    dates = data.purchase_orders[["po_id", "po_date"]]
    frame = frame.merge(dates, on="po_id", validate="one_to_one").sort_values("po_date")
    n = len(frame)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    train, valid, test = frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]
    feature_names = model_columns(features)
    categorical = [c for c in feature_names if not is_numeric_dtype(frame[c])]
    numeric = [c for c in feature_names if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    preprocessor.fit(train[feature_names])
    x_train = preprocessor.transform(train[feature_names])
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
    }
    results = {"split": {"train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
                         "train_end": str(train["po_date"].max().date()), "validation_end": str(valid["po_date"].max().date()),
                         "test_start": str(test["po_date"].min().date())}, "models": {}}
    for name, model in models.items():
        model.fit(x_train, train["risk_level"])
        results["models"][name] = {
            "validation": metrics(model, preprocessor, valid[feature_names], valid["risk_level"]),
            "test": metrics(model, preprocessor, test[feature_names], test["risk_level"]),
        }
    best_name = max(results["models"], key=lambda k: results["models"][k]["validation"]["f1_weighted"])
    best = models[best_name]
    # Refit only the selected model on all historical labels after chronological comparison.
    best.fit(preprocessor.transform(frame[feature_names]), frame["risk_level"])
    artifact = {"model": best, "preprocessor": preprocessor, "feature_builder": builder,
                "feature_names": feature_names, "model_name": best_name}
    joblib.dump(artifact, out / "riskwise_model.joblib")

    future = pd.read_csv(root / "new_po_test_cases.csv")
    truth = pd.read_csv(root / "new_po_ground_truth.csv")
    future_features = builder.build(future, future.rename(columns={"po_id": "po_id"}))
    # Future inputs intentionally contain no ground-truth columns.
    predictor = RiskPredictor(artifact)
    predictions = predictor.predict_frame(future_features)
    future_eval = predictions.merge(truth, on="po_id", validate="one_to_one")
    results["selected_model"] = best_name
    results["future_evaluation"] = {
        "rows": len(future_eval),
        "accuracy_level": round(float(accuracy_score(future_eval.ground_truth_risk_level, future_eval.risk_level)), 4),
        "f1_weighted_level": round(float(f1_score(future_eval.ground_truth_risk_level, future_eval.risk_level, average="weighted")), 4),
        "high_precision": round(float(precision_score(future_eval.ground_truth_risk_level == "HIGH", future_eval.risk_level == "HIGH", zero_division=0)), 4),
        "high_recall": round(float(recall_score(future_eval.ground_truth_risk_level == "HIGH", future_eval.risk_level == "HIGH", zero_division=0)), 4),
        "score_mae": round(float(np.abs(future_eval.risk_score - future_eval.ground_truth_risk_score).mean()), 4),
    }
    predictions.to_csv(out / "future_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="shared_data")
    parser.add_argument("--artifact-dir", default="risk_engine/artifacts")
    args = parser.parse_args()
    main(args.data_dir, args.artifact_dir)
