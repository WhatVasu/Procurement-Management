import importlib

from fastapi.testclient import TestClient


def test_submitted_purchase_order_appears_in_pipeline(monkeypatch):
    api_module = importlib.import_module("api.api")

    class FakePipeline:
        def __init__(self):
            self.engine = self

        def assess_case(self, case):
            return {
                "po_id": case["po_id"],
                "risk_score": 82.0,
                "risk_level": "HIGH",
                "fraud_score": 0.0,
                "fraud_level": "LOW",
                "fraud_flag": False,
                "primary_risk_reason": "High supplier risk",
                "primary_fraud_reason": "No fraud features supplied.",
                "overall_score": 82.0,
                "overall_status": "ESCALATE",
            }

        def run(self):
            return {
                "total_pos": 0,
                "flagged_count": 0,
                "review_count": 0,
                "approve_count": 0,
                "all_pos": [],
                "flagged_pos": [],
                "review_pos": [],
                "reasoning": {"items": []},
            }

        @staticmethod
        def _reason_about_flags(flagged):
            return {"items": [{"po_id": item["po_id"]} for item in flagged]}

    monkeypatch.setattr(api_module.RiskPredictor, "load", lambda _: object())
    monkeypatch.setattr(api_module.FraudEngine, "load", lambda _: object())
    monkeypatch.setattr(api_module, "RiskWisePipeline", FakePipeline)
    monkeypatch.setattr(api_module.memory, "retain_alert", lambda *args: True)
    client = TestClient(api_module.create_app())
    payload = {
        "po_id": "PO-NEW-001",
        "po_date": "2026-09-26",
        "supplier_id": "S001",
        "buyer_id": "B001",
        "department": "Operations",
        "currency": "USD",
        "payment_terms": "Net 30",
        "delivery_window_days": 12,
        "product_id": "P0001",
        "quantity": 120,
        "unit_price": 42.5,
    }

    submitted = client.post("/api/procurement/submit", json=payload)
    dashboard_data = client.post("/api/pipeline/run").json()

    assert submitted.status_code == 200
    assert submitted.json()["overall_status"] == "ESCALATE"
    assert submitted.json()["is_submitted"] is True
    assert dashboard_data["total_pos"] == 1
    assert dashboard_data["flagged_count"] == 1
    assert dashboard_data["all_pos"][0]["po_id"] == payload["po_id"]
    assert client.get("/procurement").status_code == 200
    assert client.post("/api/procurement/submit", json=payload).status_code == 409


def test_supplier_memory_includes_provided_historical_performance(monkeypatch):
    api_module = importlib.import_module("api.api")
    monkeypatch.setattr(api_module.memory, "recall_supplier_context", lambda _: [])
    client = TestClient(api_module.create_app())

    response = client.get("/suppliers/S001/memory", params={"as_of": "2026-09-26"})

    assert response.status_code == 200
    context = response.json()
    assert context
    assert context[0].startswith("Historical supplier performance as of 2026-09")
    assert "quality" in context[0]
    assert "prior months" in context[0]