def main():
    """Run a simple RiskWise combined assessment smoke test."""

from riskwise_engine import RiskWiseEngine


if __name__ == "__main__":
    engine = RiskWiseEngine()
    sample_po = {
   
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
  "unit_price": 25000

    }
    print(engine.assess_po(sample_po))


if __name__ == "__main__":
    main()
