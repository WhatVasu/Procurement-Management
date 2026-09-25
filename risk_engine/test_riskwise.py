from pathlib import Path

import pandas as pd

from risk_engine.audit import audit
from risk_engine.riskwise_engine import FeatureBuilder, RiskWiseData, risk_level


ROOT = Path(__file__).parents[1]
DATA = ROOT / "shared_data"


def test_audit_relationships_and_missingness():
    report = audit(DATA)
    assert report["relationships"]["orphan_po_items"] == 0
    assert report["relationships"]["orphan_suppliers"] == 0
    assert report["relationships"]["orphan_products"] == 0
    assert report["relationships"]["duplicate_po_ids"] == 0


def test_history_is_point_in_time_and_rates_present():
    data = RiskWiseData.load(DATA)
    builder = FeatureBuilder(data)
    po = data.purchase_orders.sort_values("po_date").iloc[[0]]
    features = builder.build(po)
    assert {"quality_rate", "delivery_delay_rate"}.issubset(features.columns)
    supplier = po.iloc[0].supplier_id
    cutoff = po.iloc[0].po_date.to_period("M")
    assert data.performance[(data.performance.supplier_id == supplier) & (data.performance.month >= cutoff)].empty or features.iloc[0].history_months <= 3


def test_future_inputs_have_150_rows_and_no_labels():
    cases = pd.read_csv(DATA / "new_po_test_cases.csv")
    truth = pd.read_csv(DATA / "new_po_ground_truth.csv")
    assert len(cases) == 150
    assert not set(truth.columns[1:]).intersection(cases.columns)
    assert risk_level(0) == "LOW"
    assert risk_level(35) == "MEDIUM"
    assert risk_level(70) == "HIGH"
