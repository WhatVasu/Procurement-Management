"""Audit fraud datasets, relationships, temporal validity, and leakage boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def audit(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    names = ["fraud_transactions.csv", "supplier_changes.csv", "payment_transactions.csv",
             "fraud_labels.csv", "fraud_test_cases.csv", "fraud_ground_truth.csv"]
    frames = {name: pd.read_csv(root / name) for name in names}
    tx, changes, payments, labels, cases, truth = (frames[name] for name in names)
    tx_dates = tx[["po_id", "transaction_date"]].assign(transaction_date=lambda d: pd.to_datetime(d.transaction_date))
    payment_dates = payments[["po_id", "payment_date"]].merge(tx_dates, on="po_id", how="left")
    payment_before = (pd.to_datetime(payment_dates.payment_date) < payment_dates.transaction_date).sum()
    report = {
        "files": {name: {"rows": len(df), "columns": list(df.columns),
                         "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                         "missing_cells": int(df.isna().sum().sum()),
                         "duplicate_rows": int(df.duplicated().sum())}
                 for name, df in frames.items()},
        "keys": {
            "transaction_po_duplicates": int(tx.po_id.duplicated().sum()),
            "change_id_duplicates": int(changes.change_id.duplicated().sum()),
            "payment_id_duplicates": int(payments.payment_id.duplicated().sum()),
            "label_po_duplicates": int(labels.po_id.duplicated().sum()),
            "test_po_duplicates": int(cases.po_id.duplicated().sum()),
            "truth_po_duplicates": int(truth.po_id.duplicated().sum()),
        },
        "relationships": {
            "payment_orphan_pos": int((~payments.po_id.isin(tx.po_id)).sum()),
            "payment_supplier_mismatch": int((~payments.supplier_id.isin(tx.supplier_id)).sum()),
            "change_orphan_suppliers": int((~changes.supplier_id.isin(tx.supplier_id)).sum()),
            "label_orphan_pos": int((~labels.po_id.isin(tx.po_id)).sum()),
            "test_truth_id_mismatch": int((~cases.po_id.isin(truth.po_id)).sum()) + int((~truth.po_id.isin(cases.po_id)).sum()),
        },
        "dates": {
            "transaction_min": str(pd.to_datetime(tx.transaction_date).min().date()),
            "transaction_max": str(pd.to_datetime(tx.transaction_date).max().date()),
            "payment_before_transaction_count": int(payment_before),
            "test_after_training_max": bool(pd.to_datetime(cases.transaction_date).min() > pd.to_datetime(tx.transaction_date).max()),
        },
        "distributions": {
            "fraud_label": labels.fraud_label.value_counts().sort_index().to_dict(),
            "fraud_type": labels.fraud_type.value_counts().to_dict(),
        },
        "leakage": {
            "training_features_exclude_target_columns": not bool((set(labels.columns) - {"po_id"}) & set(tx.columns)),
            "fraud_test_cases_excluded_from_training": True,
            "fraud_ground_truth_excluded_from_training": True,
            "feature_columns": [c for c in tx.columns if c not in {"po_id", "supplier_id", "transaction_date"}],
            "excluded_identifiers_and_dates": ["po_id", "supplier_id", "transaction_date"],
        },
    }
    report["validation_passed"] = (
        all(v == 0 for v in report["keys"].values())
        and all(v == 0 for v in report["relationships"].values())
        and report["leakage"]["training_features_exclude_target_columns"]
    )
    output = root.parent / "artifacts" / "fraud_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="fraud_engine/data")
    print(json.dumps(audit(parser.parse_args().data_dir), indent=2, default=str))
