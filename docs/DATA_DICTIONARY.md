# RiskWise Procurement Dataset — Data Dictionary

Historical procurement data: 2023-01-01 to 2026-08-31. Future test POs: 2026-09-01 to 2026-09-30.

| File | Column(s) | Description | Available at PO creation | Safe for prediction |
|---|---|---|---|---|
| suppliers.csv | supplier_id, supplier_name, supplier_category, country, region, supplier_tier, supplier_start_date, supplier_status | Supplier master data | YES | YES |
| products.csv | product_id, product_name, category, subcategory, department, unit_of_measure, criticality, standard_unit_price | Product master data | YES | YES |
| purchase_orders.csv | po_id, po_date, supplier_id, buyer_id, department, currency, payment_terms, delivery_window_days, shipping_method, total_po_value | One row per historical PO | YES | YES except identifiers |
| purchase_order_items.csv | po_id, line_item_id, product_id, quantity, unit_price, requested_delivery_date | PO line items; line total = quantity × unit_price | YES | YES except identifiers |
| supplier_performance.csv | supplier_id, month, total_orders, completed_orders, on_time_deliveries, late_deliveries, quality_checked_units, defective_units, cancelled_orders, disputed_orders, average_delivery_delay_days, quality_rate, delivery_delay_rate, cancellation_rate, dispute_rate | Monthly historical supplier performance | Historical only | YES if restricted to data available before/at PO |
| market_prices.csv | product_id, month, market_reference_price, market_lower_bound, market_upper_bound | Monthly product market pricing | YES for current PO month | YES |
| risk_labels.csv | po_id, risk_score, risk_level, primary_risk_reason, secondary_risk_reason | Historical training targets | NO | NO |
| new_po_test_cases.csv | po_id, po_date, supplier_id, buyer_id, department, currency, payment_terms, delivery_window_days, product_id, quantity, unit_price | Future production-like PO inputs | YES | YES |
| new_po_ground_truth.csv | po_id, ground_truth_risk_score, ground_truth_risk_level | Evaluation-only future labels | NO | NO |

Quality rate = 100 × (quality_checked_units - defective_units) / quality_checked_units.
Delivery delay rate = 100 × late_deliveries / completed_orders.

Do not feed target columns from risk_labels.csv or new_po_ground_truth.csv to the prediction model.
