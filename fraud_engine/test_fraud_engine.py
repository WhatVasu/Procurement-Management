from pathlib import Path
import pandas as pd
from .audit import audit
from .fraud_engine import FraudEngine, FraudThresholds, fraud_level, supported_reasons


ROOT = Path(__file__).parents[1]
DATA = ROOT / "fraud_engine" / "data"
ARTIFACT = ROOT / "fraud_engine" / "artifacts" / "fraud_model.joblib"


def test_audit_and_leakage_boundary():
    report = audit(DATA)
    assert report["validation_passed"]
    assert report["leakage"]["fraud_test_cases_excluded_from_training"]
    assert report["leakage"]["fraud_ground_truth_excluded_from_training"]


def test_thresholds_and_supported_reason():
    thresholds = FraudThresholds()
    assert fraud_level(0, thresholds) == "LOW"
    assert fraud_level(35, thresholds) == "MEDIUM"
    assert fraud_level(70, thresholds) == "HIGH"
    assert "price materially above market" in supported_reasons(pd.Series({"price_deviation_from_market_percent": 30}))[0]


def test_persisted_model_inference():
    engine = FraudEngine.load(ARTIFACT)
    case = pd.read_csv(DATA / "fraud_test_cases.csv").iloc[0].to_dict()
    result = engine.check(case)
    assert {"fraud_score", "fraud_level", "fraud_flag", "primary_reasons"} <= result.keys()
