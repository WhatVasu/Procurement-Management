"""FastAPI interface for RiskWise inference."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_engine.riskwise_engine import RiskPredictor
from fraud_engine.fraud_engine import FraudEngine


class PurchaseOrderRequest(BaseModel):
    po_id: str
    po_date: str
    supplier_id: str
    buyer_id: str
    department: str
    currency: str
    payment_terms: str
    delivery_window_days: float = Field(gt=0)
    product_id: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(gt=0)


class FraudCheckRequest(BaseModel):
    model_config = {"extra": "allow"}
    po_id: str
    supplier_id: str
    transaction_date: str
    total_amount: float = Field(gt=0)
    average_historical_po_value: float
    amount_deviation_percent: float
    current_unit_price: float
    average_historical_unit_price: float
    market_price: float
    price_deviation_from_market_percent: float
    quantity: float = Field(gt=0)
    average_historical_quantity: float
    quantity_deviation_percent: float
    previous_po_count_30d: int = Field(ge=0)
    previous_po_value_30d: float = Field(ge=0)
    similar_po_count_7d: int = Field(ge=0)
    duplicate_similarity_flag: int = Field(ge=0, le=1)
    split_transaction_flag: int = Field(ge=0, le=1)
    unusual_timing_flag: int = Field(ge=0, le=1)
    supplier_inactive_days_before_po: int = Field(ge=0)
    bank_account_recently_changed: int = Field(ge=0, le=1)
    address_recently_changed: int = Field(ge=0, le=1)
    contact_recently_changed: int = Field(ge=0, le=1)


def create_app(artifact_path: str | Path = "risk_engine/artifacts/riskwise_model.joblib") -> FastAPI:
    predictor = RiskPredictor.load(artifact_path)
    fraud_path = Path("fraud_engine/artifacts/fraud_model.joblib")
    fraud_predictor = FraudEngine.load(fraud_path) if fraud_path.exists() else None
    app = FastAPI(title="RiskWise Purchase Order Risk Detection Engine", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PurchaseOrderRequest) -> dict[str, Any]:
        try:
            return predictor.predict_po_risk(request.model_dump())
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/fraud/check")
    def fraud_check(request: FraudCheckRequest) -> dict[str, Any]:
        if fraud_predictor is None:
            raise HTTPException(status_code=503, detail="Fraud Detection Engine artifact is unavailable")
        try:
            return fraud_predictor.check(request.model_dump())
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app() if Path("risk_engine/artifacts/riskwise_model.joblib").exists() else FastAPI(title="RiskWise")
