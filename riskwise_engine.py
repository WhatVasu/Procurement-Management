"""Combined RiskWise purchase-order assessment layer.

This is the third module in the project: it coordinates the supplier-risk model and the
fraud model into a single operational assessment, while remaining compatible with the
existing risk_engine and fraud_engine packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_engine.fraud_engine import FraudEngine
from risk_engine import riskwise_engine as risk_engine_module
from risk_engine.riskwise_engine import RiskPredictor

FeatureBuilder = risk_engine_module.FeatureBuilder
RiskWiseData = risk_engine_module.RiskWiseData
risk_level = risk_engine_module.risk_level
score_to_level = risk_engine_module.score_to_level
model_columns = risk_engine_module.model_columns

DEFAULT_RISK_ARTIFACT = Path("risk_engine/artifacts/riskwise_model.joblib")
DEFAULT_FRAUD_ARTIFACT = Path("fraud_engine/artifacts/fraud_model.joblib")
PO_INPUT_FIELDS = (
    "po_id", "po_date", "supplier_id", "buyer_id", "department", "currency",
    "payment_terms", "delivery_window_days", "product_id", "quantity", "unit_price",
)


class RiskWiseEngine:
    """Aggregate PO risk and fraud signals into a single recommendation."""

    def __init__(
        self,
        risk_artifact: str | Path = DEFAULT_RISK_ARTIFACT,
        fraud_artifact: str | Path = DEFAULT_FRAUD_ARTIFACT,
    ) -> None:
        self.risk_artifact = Path(risk_artifact)
        self.fraud_artifact = Path(fraud_artifact)
        self.risk_predictor = RiskPredictor.load(self.risk_artifact) if self.risk_artifact.exists() else None
        self.fraud_engine = FraudEngine.load(self.fraud_artifact) if self.fraud_artifact.exists() else None

    @classmethod
    def load(cls, risk_artifact: str | Path = DEFAULT_RISK_ARTIFACT, fraud_artifact: str | Path = DEFAULT_FRAUD_ARTIFACT) -> "RiskWiseEngine":
        return cls(risk_artifact=risk_artifact, fraud_artifact=fraud_artifact)

    def _empty_fraud_result(self) -> dict[str, Any]:
        return {
            "fraud_score": 0.0,
            "fraud_level": "LOW",
            "fraud_flag": False,
            "primary_reasons": "No fraud features supplied.",
        }

    def assess_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """Route the PO and fraud feature subsets from one combined test case."""
        missing_po_fields = [
            name for name in PO_INPUT_FIELDS
            if name not in case or pd.isna(case[name])
        ]
        if missing_po_fields:
            raise ValueError(f"Missing purchase-order fields: {', '.join(missing_po_fields)}")

        purchase_order = {name: case[name] for name in PO_INPUT_FIELDS}
        fraud_transaction = None
        if self.fraud_engine is not None:
            feature_names = self.fraud_engine.feature_names
            fraud_only_fields = set(feature_names).difference(PO_INPUT_FIELDS)
            supplied_fraud_fields = [
                name for name in feature_names
                if name in fraud_only_fields and name in case and not pd.isna(case[name])
            ]
            if supplied_fraud_fields:
                missing_fraud_fields = [
                    name for name in feature_names
                    if name not in case or pd.isna(case[name])
                ]
                if missing_fraud_fields:
                    raise ValueError(
                        f"Missing fraud feature fields: {', '.join(missing_fraud_fields)}"
                    )
                context_fields = ("po_id", "supplier_id", "transaction_date")
                fraud_transaction = {
                    name: case[name]
                    for name in (*context_fields, *feature_names)
                    if name in case and not pd.isna(case[name])
                }
        elif "transaction_date" in case and not pd.isna(case["transaction_date"]):
            raise FileNotFoundError(f"Fraud artifact not found at {self.fraud_artifact}")

        return self.assess_po(purchase_order, fraud_transaction)

    def assess_po(
        self,
        purchase_order: dict[str, Any],
        fraud_transaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.risk_predictor is None:
            raise FileNotFoundError(f"Risk artifact not found at {self.risk_artifact}")

        risk_result = self.risk_predictor.predict_po_risk(purchase_order)
        fraud_result = self._empty_fraud_result()

        if fraud_transaction is not None:
            if self.fraud_engine is None:
                raise FileNotFoundError(f"Fraud artifact not found at {self.fraud_artifact}")
            fraud_result = self.fraud_engine.check(fraud_transaction)

        risk_score = float(risk_result.get("risk_score", 0.0))
        fraud_score = float(fraud_result.get("fraud_score", 0.0))
        overall_score = risk_score if fraud_transaction is None else (risk_score + fraud_score) / 2.0
        overall_score = round(overall_score, 1)

        risk_level_value = risk_result.get("risk_level", "LOW")
        fraud_level_value = fraud_result.get("fraud_level", "LOW")
        fraud_flag = bool(fraud_result.get("fraud_flag", False))
        if risk_level_value == "HIGH" or fraud_level_value == "HIGH" or fraud_flag:
            overall_status = "ESCALATE"
        elif overall_score >= 40 or risk_level_value == "MEDIUM" or fraud_level_value == "MEDIUM":
            overall_status = "REVIEW"
        else:
            overall_status = "APPROVE"

        return {
            "po_id": risk_result.get("po_id", purchase_order.get("po_id")),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level_value,
            "fraud_score": round(fraud_score, 1),
            "fraud_level": fraud_level_value,
            "fraud_flag": fraud_flag,
            "primary_risk_reason": risk_result.get("primary_risk_reason", "No supplier-risk drivers detected."),
            "primary_fraud_reason": fraud_result.get("primary_reasons", "No fraud indicators detected."),
            "overall_score": overall_score,
            "overall_status": overall_status,
        }


__all__ = [
    "RiskWiseEngine",
    "FeatureBuilder",
    "RiskWiseData",
    "RiskPredictor",
    "risk_level",
    "score_to_level",
    "model_columns",
    "FraudEngine",
]


