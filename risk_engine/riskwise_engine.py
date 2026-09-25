"""RiskWise purchase-order risk feature engineering and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


LEVELS = ("LOW", "MEDIUM", "HIGH")
TARGET_COLUMNS = {"po_id", "risk_score", "risk_level", "primary_risk_reason", "secondary_risk_reason"}


def _month(value: Any) -> pd.Period:
    return pd.Period(pd.Timestamp(value), freq="M")


def risk_level(score: float) -> str:
    if score < 35:
        return "LOW"
    if score < 70:
        return "MEDIUM"
    return "HIGH"


def score_to_level(score: float) -> str:
    return risk_level(float(np.clip(score, 0, 100)))


@dataclass
class RiskWiseData:
    root: Path
    suppliers: pd.DataFrame
    products: pd.DataFrame
    purchase_orders: pd.DataFrame
    items: pd.DataFrame
    performance: pd.DataFrame
    market_prices: pd.DataFrame

    @classmethod
    def load(cls, root: str | Path) -> "RiskWiseData":
        root = Path(root)
        read = lambda name: pd.read_csv(root / name)
        data = cls(
            root,
            read("suppliers.csv"),
            read("products.csv"),
            read("purchase_orders.csv"),
            read("purchase_order_items.csv"),
            read("supplier_performance.csv"),
            read("market_prices.csv"),
        )
        data.purchase_orders["po_date"] = pd.to_datetime(data.purchase_orders["po_date"])
        data.items["requested_delivery_date"] = pd.to_datetime(data.items["requested_delivery_date"])
        data.suppliers["supplier_start_date"] = pd.to_datetime(data.suppliers["supplier_start_date"])
        data.performance["month"] = data.performance["month"].map(lambda x: pd.Period(x, freq="M"))
        data.market_prices["month"] = data.market_prices["month"].map(lambda x: pd.Period(x, freq="M"))
        return data


class FeatureBuilder:
    """Builds creation-time PO features; history is strictly before the PO month."""

    def __init__(self, data: RiskWiseData):
        self.data = data
        self.products = data.products.set_index("product_id")
        self.suppliers = data.suppliers.set_index("supplier_id")
        self.performance = data.performance
        self.market = data.market_prices.set_index(["product_id", "month"])

    def _history(self, supplier_id: str, po_month: pd.Period) -> dict[str, float]:
        hist = self.performance[
            (self.performance["supplier_id"] == supplier_id)
            & (self.performance["month"] < po_month)
        ].sort_values("month").tail(3)
        if hist.empty:
            return {
                "quality_rate": 100.0, "delivery_delay_rate": 0.0,
                "cancellation_rate": 0.0, "dispute_rate": 0.0,
                "average_delivery_delay_days": 0.0, "history_months": 0.0,
            }
        weights = np.arange(1, len(hist) + 1, dtype=float)
        def weighted(column: str) -> float:
            return float(np.average(hist[column].astype(float), weights=weights))
        return {
            "quality_rate": weighted("quality_rate"),
            "delivery_delay_rate": weighted("delivery_delay_rate"),
            "cancellation_rate": weighted("cancellation_rate"),
            "dispute_rate": weighted("dispute_rate"),
            "average_delivery_delay_days": weighted("average_delivery_delay_days"),
            "history_months": float(len(hist)),
        }

    def _market(self, product_id: str, po_month: pd.Period) -> dict[str, float]:
        key = (product_id, po_month)
        if key in self.market.index:
            row = self.market.loc[key]
        else:
            prior = self.data.market_prices[
                (self.data.market_prices["product_id"] == product_id)
                & (self.data.market_prices["month"] <= po_month)
            ].sort_values("month")
            row = prior.iloc[-1] if not prior.empty else None
        if row is None:
            return {"market_reference_price": 0.0, "market_lower_bound": 0.0, "market_upper_bound": 0.0}
        return {k: float(row[k]) for k in ("market_reference_price", "market_lower_bound", "market_upper_bound")}

    def build(self, pos: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return one feature row per PO without consuming risk labels."""
        pos = pos.copy()
        pos["po_date"] = pd.to_datetime(pos["po_date"])
        items = self.data.items if items is None else items.copy()
        if "requested_delivery_date" in items:
            items["requested_delivery_date"] = pd.to_datetime(items["requested_delivery_date"])
        item_groups = items.groupby("po_id", sort=False)
        rows: list[dict[str, Any]] = []
        for po in pos.to_dict("records"):
            po_id = po["po_id"]
            lines = item_groups.get_group(po_id) if po_id in item_groups.groups else pd.DataFrame()
            quantity = float(lines["quantity"].sum()) if not lines.empty else float(po.get("quantity", 0))
            value = float((lines["quantity"] * lines["unit_price"]).sum()) if not lines.empty else float(po.get("quantity", 0) * po.get("unit_price", 0))
            prices = lines["unit_price"].astype(float).to_numpy() if not lines.empty else np.array([float(po.get("unit_price", 0))])
            product_id = str(lines.iloc[0]["product_id"]) if not lines.empty else str(po["product_id"])
            product = self.products.loc[product_id]
            market = self._market(product_id, _month(po["po_date"]))
            reference = market["market_reference_price"] or float(product["standard_unit_price"])
            supplier = self.suppliers.loc[po["supplier_id"]]
            history = self._history(po["supplier_id"], _month(po["po_date"]))
            requested_days = (
                float((lines["requested_delivery_date"] - pd.Timestamp(po["po_date"])).dt.days.mean())
                if not lines.empty and "requested_delivery_date" in lines else float(po["delivery_window_days"])
            )
            rows.append({
                "po_id": po_id,
                "supplier_id": po["supplier_id"],
                "product_id": product_id,
                "department": po["department"],
                "currency": po["currency"],
                "payment_terms": po["payment_terms"],
                "buyer_id": po["buyer_id"],
                "supplier_category": supplier["supplier_category"],
                "supplier_country": supplier["country"],
                "supplier_region": supplier["region"],
                "supplier_tier": supplier["supplier_tier"],
                "supplier_status": supplier["supplier_status"],
                "product_category": product["category"],
                "product_subcategory": product["subcategory"],
                "product_department": product["department"],
                "criticality": product["criticality"],
                "unit_of_measure": product["unit_of_measure"],
                "delivery_window_days": float(po["delivery_window_days"]),
                "requested_delivery_days": requested_days,
                "quantity": quantity,
                "total_po_value": value if value else float(po.get("total_po_value", 0)),
                "line_count": float(len(lines) or 1),
                "mean_unit_price": float(np.mean(prices)),
                "max_unit_price": float(np.max(prices)),
                "price_vs_market_pct": float((np.mean(prices) - reference) / reference * 100) if reference else 0.0,
                "market_reference_price": reference,
                "market_lower_bound": market["market_lower_bound"],
                "market_upper_bound": market["market_upper_bound"],
                "supplier_tenure_days": float((pd.Timestamp(po["po_date"]) - supplier["supplier_start_date"]).days),
                "po_month": str(_month(po["po_date"])),
                **history,
            })
        return pd.DataFrame(rows)


def model_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"po_id", "supplier_id", "product_id", "po_month"}]


class RiskPredictor:
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["model"]
        self.feature_builder = artifact["feature_builder"]
        self.feature_names = artifact["feature_names"]

    @classmethod
    def load(cls, path: str | Path) -> "RiskPredictor":
        return cls(joblib.load(path))

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        x = features[self.feature_names]
        probabilities = self.model.predict_proba(self.preprocessor.transform(x))
        classes = list(self.model.classes_)
        high_index = classes.index("HIGH")
        predicted = self.model.predict(self.preprocessor.transform(x))
        output = []
        for i, row in features.reset_index(drop=True).iterrows():
            reasons = explain_features(row)
            score = float(probabilities[i, high_index] * 100)
            output.append({
                "po_id": row["po_id"],
                "risk_score": round(score, 1),
                "risk_level": score_to_level(score),
                "quality_rate": round(float(row["quality_rate"]), 2),
                "delivery_delay_rate": round(float(row["delivery_delay_rate"]), 2),
                "primary_risk_reason": reasons[0],
                "secondary_risk_reason": reasons[1],
            })
        return pd.DataFrame(output)

    def predict_po_risk(self, po: dict[str, Any]) -> dict[str, Any]:
        po_df = pd.DataFrame([po])
        if "product_id" not in po_df or "quantity" not in po_df or "unit_price" not in po_df:
            raise ValueError("Request requires product_id, quantity, and unit_price")
        items = pd.DataFrame([{
            "po_id": po["po_id"], "product_id": po["product_id"], "quantity": po["quantity"],
            "unit_price": po["unit_price"], "requested_delivery_date": pd.Timestamp(po["po_date"]) + pd.Timedelta(days=po["delivery_window_days"]),
        }])
        return self.predict_frame(self.feature_builder.build(po_df, items)).iloc[0].to_dict()


def explain_features(row: pd.Series) -> tuple[str, str]:
    reasons: list[tuple[float, str]] = []
    if row["delivery_delay_rate"] >= 25:
        reasons.append((row["delivery_delay_rate"], "Supplier has above-normal delivery delay"))
    if row["quality_rate"] < 95:
        reasons.append((100 - row["quality_rate"], "Supplier quality performance is below preferred range"))
    if abs(row["price_vs_market_pct"]) >= 15:
        reasons.append((abs(row["price_vs_market_pct"]), "PO price is significantly different from market reference price"))
    if row["cancellation_rate"] >= 8:
        reasons.append((row["cancellation_rate"], "Supplier has elevated cancellation rate"))
    if row["dispute_rate"] >= 5:
        reasons.append((row["dispute_rate"], "Supplier has elevated dispute rate"))
    if row["line_count"] >= 5:
        reasons.append((row["line_count"], "PO has high item complexity"))
    if row["delivery_window_days"] <= 10:
        reasons.append((20 - row["delivery_window_days"], "PO has a short delivery window"))
    reasons.sort(key=lambda x: (-x[0], x[1]))
    selected = [r[1] for r in reasons[:2]]
    while len(selected) < 2:
        selected.append("Recent supplier performance is within normal ranges")
    return selected[0], selected[1]
