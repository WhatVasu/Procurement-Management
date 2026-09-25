"""Fraud Detection Engine: feature-only inference over fraud transaction records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class FraudThresholds:
    medium_score: float = 35.0
    high_score: float = 70.0
    flag_score: float = 70.0

    def __post_init__(self) -> None:
        if not (0 <= self.medium_score <= self.high_score <= 100):
            raise ValueError("Thresholds must satisfy 0 <= medium_score <= high_score <= 100")
        if not 0 <= self.flag_score <= 100:
            raise ValueError("flag_score must be between 0 and 100")


def fraud_level(score: float, thresholds: FraudThresholds = FraudThresholds()) -> str:
    if score < thresholds.medium_score:
        return "LOW"
    if score < thresholds.high_score:
        return "MEDIUM"
    return "HIGH"


def supported_reasons(row: pd.Series) -> list[str]:
    candidates: list[tuple[float, str]] = []
    checks = [
        ("price_deviation_from_market_percent", 25, "price materially above market"),
        ("amount_deviation_percent", 50, "transaction amount materially differs from supplier history"),
        ("quantity_deviation_percent", 75, "unusual quantity versus supplier history"),
        ("duplicate_similarity_flag", 0, "near-duplicate transaction"),
        ("split_transaction_flag", 0, "split-transaction pattern"),
        ("unusual_timing_flag", 0, "unusual transaction timing"),
        ("bank_account_recently_changed", 0, "recent bank-account change"),
        ("address_recently_changed", 0, "recent address change"),
        ("contact_recently_changed", 0, "recent contact change"),
    ]
    for column, threshold, reason in checks:
        value = float(row.get(column, 0))
        if column.endswith("_flag") or column.endswith("_changed"):
            active, weight = value == 1, 100 if value == 1 else 0
        elif column == "price_deviation_from_market_percent":
            active, weight = value >= threshold, max(0.0, value)
        else:
            active, weight = abs(value) >= threshold, abs(value)
        if active:
            candidates.append((weight, reason))
    if float(row.get("previous_po_count_30d", 0)) >= 5:
        candidates.append((float(row["previous_po_count_30d"]), "high recent supplier transaction frequency"))
    if float(row.get("similar_po_count_7d", 0)) >= 2:
        candidates.append((float(row["similar_po_count_7d"]), "multiple similar recent transactions"))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [reason for _, reason in candidates[:3]] or ["No sufficient combination of fraud indicators."]


class FraudEngine:
    def __init__(self, artifact: dict[str, Any]):
        self.model = artifact["model"]
        self.preprocessor = artifact["preprocessor"]
        self.feature_names = artifact["feature_names"]
        self.thresholds = FraudThresholds(**artifact["thresholds"])

    @classmethod
    def load(cls, path: str | Path) -> "FraudEngine":
        return cls(joblib.load(path))

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.preprocessor.transform(frame[self.feature_names])
        probability = self.model.predict_proba(transformed)
        positive_index = list(self.model.classes_).index(1)
        rows = []
        for i, source in frame.reset_index(drop=True).iterrows():
            score = float(probability[i, positive_index] * 100)
            rows.append({
                "po_id": source["po_id"],
                "fraud_score": round(score, 1),
                "fraud_level": fraud_level(score, self.thresholds),
                "fraud_flag": bool(score >= self.thresholds.flag_score),
                "primary_reasons": "; ".join(supported_reasons(source)),
            })
        return pd.DataFrame(rows)

    def check(self, transaction: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([transaction])
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"Missing fraud feature fields: {', '.join(missing)}")
        return self.predict_frame(frame).iloc[0].to_dict()
