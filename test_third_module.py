from types import SimpleNamespace

import pytest

from riskwise_engine import RiskWiseEngine


def test_riskwise_engine_combines_risk_and_fraud_scores():
    engine = RiskWiseEngine()
    po = {
        "po_id": "PO-1001",
        "po_date": "2024-03-15",
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

    result = engine.assess_po(po)

    assert result["risk_score"] >= 0
    assert result["fraud_score"] >= 0
    assert result["overall_score"] >= 0
    assert result["overall_status"] in {"APPROVE", "REVIEW", "ESCALATE"}


def test_assess_case_routes_combined_schema_to_both_models():
    engine = RiskWiseEngine()
    case = {
        "po_id": "TEST-MANUAL-001",
        "po_date": "2026-09-23",
        "supplier_id": "S083",
        "buyer_id": "B007",
        "department": "Engineering",
        "currency": "INR",
        "payment_terms": "Net 30",
        "delivery_window_days": 10,
        "product_id": "P0039",
        "quantity": 500,
        "unit_price": 25000,
        "transaction_date": "2023-01-15",
        "total_amount": 31866.72528656753,
        "average_historical_po_value": 13357.33,
        "amount_deviation_percent": 138.57107136356984,
        "current_unit_price": 21.204747060964,
        "average_historical_unit_price": 1377.17,
        "market_price": 14.27,
        "price_deviation_from_market_percent": 48.596685781107226,
        "average_historical_quantity": 25,
        "quantity_deviation_percent": 200.221634898591,
        "previous_po_count_30d": 0,
        "previous_po_value_30d": 0,
        "similar_po_count_7d": 0,
        "duplicate_similarity_flag": 1,
        "split_transaction_flag": 1,
        "unusual_timing_flag": 1,
        "supplier_inactive_days_before_po": 999,
        "bank_account_recently_changed": 1,
        "address_recently_changed": 0,
        "contact_recently_changed": 0,
    }

    result = engine.assess_case(case)

    assert result["po_id"] == case["po_id"]
    assert result["fraud_score"] >= 0
    assert result["fraud_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["fraud_flag"] == (
        result["fraud_score"] >= engine.fraud_engine.thresholds.flag_score
    )
    assert result["overall_status"] in {"APPROVE", "REVIEW", "ESCALATE"}


def test_assess_case_keeps_legacy_po_only_rows_supplier_only():
    engine = RiskWiseEngine()
    case = {
        "po_id": "TEST-LEGACY-001",
        "po_date": "2026-09-23",
        "supplier_id": "S083",
        "buyer_id": "B007",
        "department": "Engineering",
        "currency": "INR",
        "payment_terms": "Net 30",
        "delivery_window_days": 10,
        "product_id": "P0039",
        "quantity": 500,
        "unit_price": 25000,
    }

    result = engine.assess_case(case)

    assert result["po_id"] == case["po_id"]
    assert result["primary_fraud_reason"] == "No fraud features supplied."


def test_assess_case_rejects_partial_fraud_features():
    engine = RiskWiseEngine()
    case = {
        "po_id": "TEST-PARTIAL-001",
        "po_date": "2026-09-23",
        "supplier_id": "S083",
        "buyer_id": "B007",
        "department": "Engineering",
        "currency": "INR",
        "payment_terms": "Net 30",
        "delivery_window_days": 10,
        "product_id": "P0039",
        "quantity": 500,
        "unit_price": 25000,
        "total_amount": 1000,
    }

    with pytest.raises(ValueError, match="Missing fraud feature fields"):
        engine.assess_case(case)


def test_high_fraud_threshold_escalates_despite_low_risk_score():
    engine = RiskWiseEngine()

    class LowRiskPredictor:
        @staticmethod
        def predict_po_risk(purchase_order):
            return {
                "po_id": purchase_order["po_id"],
                "risk_score": 10,
                "risk_level": "LOW",
            }

    class HighFraudEngine:
        thresholds = SimpleNamespace(flag_score=70)

        @staticmethod
        def check(transaction):
            return {
                "fraud_score": 90,
                "fraud_level": "HIGH",
                "fraud_flag": True,
                "primary_reasons": "near-duplicate transaction",
            }

    engine.risk_predictor = LowRiskPredictor()
    engine.fraud_engine = HighFraudEngine()

    result = engine.assess_po({"po_id": "TEST-HIGH-FRAUD"}, {})

    assert result["overall_score"] == 50
    assert result["overall_status"] == "ESCALATE"
