# RiskWise Procurement Dataset — Generation Guide

Synthetic data generated with random seed 42.

- 150 suppliers with varied categories, countries, tiers, onboarding dates and latent performance profiles.
- 300 products with categories, criticality and baseline prices.
- Monthly market prices from 2023-01 through 2026-09 with drift and occasional shocks.
- Monthly supplier performance from each supplier's latent quality/delivery behavior, with volatility and trends.
- 15,000 historical POs with 1–8 line items, varied departments, currencies, payment terms, shipping methods, quantities and delivery windows.
- Unit prices are mostly near market but include moderately and highly unusual cases.
- Risk labels use a nonlinear latent process combining recent supplier quality, delay, cancellations, disputes, price deviation, PO size, complexity, delivery window and supplier history, with interactions and noise.
- Historical label features use supplier performance from the previous three available months only, avoiding future performance leakage.
- September 2026 POs are held out as unseen future test cases.
- `new_po_ground_truth.csv` is evaluation-only and must not be supplied to the prediction engine.

Recommended model preparation: aggregate PO-line and supplier-history features using only information available at the PO creation timestamp, then use `risk_labels.csv` as the target.
