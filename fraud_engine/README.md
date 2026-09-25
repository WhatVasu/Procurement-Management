# Fraud Detection Engine

Fraud-owned datasets and documentation live in `fraud_engine/data`; the shared procurement
CSV files are not copied or modified. Training uses only `fraud_transactions.csv` joined to
`fraud_labels.csv`. `fraud_test_cases.csv` and `fraud_ground_truth.csv` are loaded only for
post-training external evaluation.

```powershell
python -m fraud_engine.audit --data-dir fraud_engine/data
python -m fraud_engine.train --data-dir fraud_engine/data --artifact-dir fraud_engine/artifacts
python -m fraud_engine.evaluate --data-dir fraud_engine/data --artifact-dir fraud_engine/artifacts
python -m pytest risk_engine/test_riskwise.py fraud_engine/test_fraud_engine.py -q
python -m uvicorn api.api:app --host 127.0.0.1 --port 8000
```

The default score is the fitted model's fraud probability multiplied by 100. Defaults are
LOW `<35`, MEDIUM `35-<70`, HIGH `>=70`, and `fraud_flag` when score is at least 70. These
are persisted in the artifact and represented by `FraudThresholds`; callers can use custom
thresholds when constructing an engine. Reasons are deterministic and based only on supported
feature signals such as price/amount/quantity deviations, duplicate/split/timing indicators,
and recent supplier profile changes.
