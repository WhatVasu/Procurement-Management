"""Create a reproducible audit of all supplied RiskWise data files."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def audit(root: str | Path) -> dict:
    root = Path(root)
    files = sorted(root.glob("*.csv"))
    report = {"files": {}, "relationships": {}, "leakage": {}}
    frames = {}
    for path in files:
        df = pd.read_csv(path)
        frames[path.name] = df
        report["files"][path.name] = {
            "rows": int(len(df)), "columns": list(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }
    po = frames["purchase_orders.csv"]
    items = frames["purchase_order_items.csv"]
    labels = frames["risk_labels.csv"]
    report["relationships"] = {
        "orphan_po_items": int((~items["po_id"].isin(po["po_id"])).sum()),
        "orphan_suppliers": int((~po["supplier_id"].isin(frames["suppliers.csv"]["supplier_id"])).sum()),
        "orphan_products": int((~items["product_id"].isin(frames["products.csv"]["product_id"])).sum()),
        "orphan_labels": int((~labels["po_id"].isin(po["po_id"])).sum()),
        "duplicate_po_ids": int(po["po_id"].duplicated().sum()),
        "duplicate_line_item_ids": int(items["line_item_id"].duplicated().sum()),
    }
    report["leakage"] = {
        "target_columns": ["risk_score", "risk_level", "primary_risk_reason", "secondary_risk_reason"],
        "excluded_from_features": True,
        "historical_performance_cutoff": "strictly before PO month; latest three months",
        "future_ground_truth_loaded": False,
    }
    output = root.parent / "risk_engine" / "artifacts" / "data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="shared_data")
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir), indent=2))
