# RiskWise Purchase Order Risk Detection Engine

This implementation is rebuilt for the supplied schema. It never uses `risk_labels.csv` or
`new_po_ground_truth.csv` as prediction inputs. Historical supplier features are weighted
averages of the three latest performance months strictly before the PO month; current-month
market prices and PO/product/supplier master data are creation-time inputs.

## Run

```powershell
python -m risk_engine.audit --data-dir shared_data
python -m risk_engine.train --data-dir shared_data --artifact-dir risk_engine/artifacts
python -m risk_engine.evaluate --data-dir shared_data --artifact-dir risk_engine/artifacts
python -m pytest risk_engine/test_riskwise.py -q
python -m uvicorn api.api:app --host 127.0.0.1 --port 8000
```

`risk_engine.train` writes `risk_engine/artifacts/riskwise_model.joblib`, `metrics.json`, and
`future_predictions.csv`. It compares Logistic Regression, Random Forest, and Extra Trees
using chronological 70/15/15 splits, then refits the best validation-F1 model on historical
labels. It evaluates all 150 future cases only after prediction.

## Scoring and reasons

The persisted classifier's calibrated class probability for HIGH is exposed as a reproducible
0-100 risk score. Levels are LOW `<35`, MEDIUM `35-<70`, and HIGH `>=70` (the same thresholds
are used by `risk_level`). The returned level is derived from this score, rather than from a
separate class-label shortcut. Reasons are deterministic, feature-supported rules sorted by
severity: delivery delay >=25%, quality <95%, price deviation >=15%, cancellation >=8%,
dispute >=5%, five or more lines, and delivery windows <=10 days. The response always includes
`quality_rate` and `delivery_delay_rate`.

## API request

`POST /predict` accepts the fields in `new_po_test_cases.csv` and returns `po_id`, `risk_score`,
`risk_level`, the two supplier rates, and two deterministic risk reasons. `GET /health` is
available for smoke tests.
