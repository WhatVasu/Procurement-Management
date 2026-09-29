"""FastAPI interface for RiskWise inference."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from datetime import date, datetime, timezone

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from risk_engine.riskwise_engine import RiskPredictor
from risk_engine import memory
from fraud_engine.fraud_engine import FraudEngine
from pipeline_orchestrator import RiskWisePipeline

ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "dashboard" / "index.html"
PROCUREMENT_UI_PATH = ROOT / "dashboard" / "procurement.html"


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


class OutcomeRequest(BaseModel):
    supplier_id: str
    outcome: str = Field(min_length=1)
    related_action: str | None = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    if memory.MODE == "embedded":
        memory.start_embedded_server()
    try:
        yield
    finally:
        if memory.MODE == "embedded":
            memory.stop_embedded_server()


def create_app(artifact_path: str | Path = "risk_engine/artifacts/riskwise_model.joblib") -> FastAPI:
    predictor = RiskPredictor.load(artifact_path)
    fraud_path = Path("fraud_engine/artifacts/fraud_model.joblib")
    fraud_predictor = FraudEngine.load(fraud_path) if fraud_path.exists() else None
    pipeline = RiskWisePipeline()
    submitted_pos: dict[str, dict[str, Any]] = {}
    app = FastAPI(
        title="RiskWise Purchase Order Risk Detection Engine",
        version="1.0",
        lifespan=app_lifespan,
    )

    def retain_alert_in_background(background_tasks: BackgroundTasks, item: dict[str, Any]) -> None:
        reasons = [
            reason for reason in (
                item.get("primary_risk_reason"), item.get("primary_fraud_reason")
            ) if reason
        ]
        background_tasks.add_task(
            memory.retain_alert,
            item.get("supplier_id", "unknown"),
            None,
            item.get("po_id", "unknown"),
            item.get("risk_level", item.get("fraud_level", "HIGH")),
            float(item.get("risk_score", item.get("overall_score", 0))),
            reasons,
            "Review flagged purchase order",
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        if not memory.ENABLED:
            memory_status = "disabled"
        else:
            client = memory._get_client()
            if client is None:
                memory_status = "unreachable"
            else:
                try:
                    client.recall(bank_id=memory.BANK_ID, query="RiskWise health check")
                    memory_status = "up"
                except Exception:
                    memory_status = "unreachable"
        return {"status": "ok", "memory": memory_status}

    @app.get("/suppliers/{supplier_id}/memory")
    def supplier_memory(supplier_id: str, as_of: date | None = None) -> list[str]:
        cutoff = pd.Period(as_of or date.today(), freq="M")
        history = predictor.feature_builder._history(supplier_id, cutoff)
        context = []
        if history["history_months"]:
            month_count = int(history["history_months"])
            context.append(
                f"Historical supplier performance as of {cutoff}: "
                f"weighted averages across {month_count} prior months: "
                f"quality {history['quality_rate']:.1f}%, "
                f"delivery delays {history['delivery_delay_rate']:.1f}%, "
                f"cancellations {history['cancellation_rate']:.1f}%, "
                f"disputes {history['dispute_rate']:.1f}%, "
                f"average delay {history['average_delivery_delay_days']:.1f} days."
            )
        context.extend(memory.recall_supplier_context(supplier_id))
        return context

    @app.post("/outcomes", status_code=202)
    def record_outcome(request: OutcomeRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
        background_tasks.add_task(
            memory.retain_outcome,
            request.supplier_id,
            None,
            request.outcome,
            request.related_action,
        )
        return {"status": "queued"}

    @app.get("/api/pipeline/run")
    @app.post("/api/pipeline/run")
    def run_pipeline() -> dict[str, Any]:
        try:
            result = pipeline.run()
            if submitted_pos:
                all_pos = [*result["all_pos"], *submitted_pos.values()]
                flagged = [
                    item for item in all_pos
                    if item["overall_status"] == "ESCALATE" or item["risk_level"] == "HIGH"
                ]
                result.update(
                    total_pos=len(all_pos),
                    flagged_count=len(flagged),
                    review_count=sum(item["overall_status"] == "REVIEW" for item in all_pos),
                    approve_count=sum(item["overall_status"] == "APPROVE" for item in all_pos),
                    all_pos=all_pos,
                    flagged_pos=flagged,
                    review_pos=[item for item in all_pos if item["overall_status"] == "REVIEW"],
                    reasoning=pipeline._reason_about_flags(flagged),
                )
            return result
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(UI_PATH)

    @app.get("/procurement", include_in_schema=False)
    def procurement() -> FileResponse:
        return FileResponse(PROCUREMENT_UI_PATH)

    @app.post("/api/procurement/submit")
    def submit_purchase_order(
        request: PurchaseOrderRequest, background_tasks: BackgroundTasks
    ) -> dict[str, Any]:
        if request.po_id in submitted_pos:
            raise HTTPException(status_code=409, detail="A purchase order with this ID was already submitted")
        try:
            result = pipeline.engine.assess_case(request.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result.update(
            is_submitted=True,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            po_date=request.po_date,
            supplier_id=request.supplier_id,
            buyer_id=request.buyer_id,
            department=request.department,
            currency=request.currency,
            payment_terms=request.payment_terms,
            delivery_window_days=request.delivery_window_days,
            product_id=request.product_id,
            quantity=request.quantity,
            unit_price=request.unit_price,
        )
        submitted_pos[request.po_id] = result
        if result["overall_status"] == "ESCALATE" or result["risk_level"] == "HIGH":
            retain_alert_in_background(background_tasks, result)
        return result

    @app.post("/predict")
    def predict(
        request: PurchaseOrderRequest, background_tasks: BackgroundTasks
    ) -> dict[str, Any]:
        try:
            result = predictor.predict_po_risk(request.model_dump())
            if result["risk_level"] == "HIGH":
                retain_alert_in_background(
                    background_tasks, {**result, "supplier_id": request.supplier_id}
                )
            return result
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

    # TODO(Risk OS): mirror persisted alerts from the dashboard batch when an
    # alert record is created instead of recomputing static test-case flags.
    # TODO(Risk OS): persist outcomes in the system of record when one is available.
    # TODO(Risk OS): retain reviewer decisions and attach fraud memory hints when
    # persistent review queues and approve/dismiss/escalate handlers are added.
    # TODO(Risk OS): mirror actions_log rows and ingested external events when
    # those system-of-record workflows exist.

    return app


app = create_app() if Path("risk_engine/artifacts/riskwise_model.joblib").exists() else FastAPI(title="RiskWise")
