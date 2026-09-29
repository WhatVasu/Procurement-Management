import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import sys
from threading import Barrier, get_ident, local
from types import ModuleType

from fastapi.testclient import TestClient

from risk_engine import memory


def test_disabled_memory_functions_return_neutral_values(monkeypatch):
    monkeypatch.setattr(memory, "ENABLED", False)
    monkeypatch.setattr(memory, "_client", None)
    monkeypatch.setattr(memory, "_client_failed", False)

    assert memory.retain("content", "context") is False
    assert memory.recall("query") == []
    assert memory.reflect("query") == ""
    assert memory.retain_alert("S1", None, "PO1", "HIGH", 80, [], "Hold PO") is False
    assert memory.retain_human_decision("S1", None, "A1", "approve", "reviewer") is False
    assert memory.retain_fraud_review("S1", None, "T1", [], "HIGH", "dismiss") is False
    assert memory.retain_action("S1", None, "hold PO", "executed") is False
    assert memory.retain_outcome("S1", None, "resolved") is False
    assert memory.retain_external_event("S1", None, "recall", "Product recalled") is False
    assert memory.recall_supplier_context("S1") == []
    assert memory.recall_similar_flags("S1", ["duplicate"]) == []
    assert memory.reflect_on_supplier("S1") == ""
    assert memory.format_for_prompt([]) == ""


def test_client_errors_are_neutral(monkeypatch):
    class RaisingClient:
        def retain(self, **kwargs):
            raise RuntimeError("offline")

        def recall(self, **kwargs):
            raise RuntimeError("offline")

        def reflect(self, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "_client", RaisingClient())
    monkeypatch.setattr(memory, "_client_failed", False)

    assert memory.retain("content", "context") is False
    assert memory.recall("query") == []
    assert memory.reflect("query") == ""
    assert memory.retain_alert("S1", None, "PO1", "HIGH", 80, [], "Hold PO") is False
    assert memory.retain_human_decision("S1", None, "A1", "approve", "reviewer") is False
    assert memory.retain_fraud_review("S1", None, "T1", [], "HIGH", "dismiss") is False
    assert memory.retain_action("S1", None, "hold PO", "executed") is False
    assert memory.retain_outcome("S1", None, "resolved") is False
    assert memory.retain_external_event("S1", None, "recall", "Product recalled") is False
    assert memory.recall_supplier_context("S1") == []
    assert memory.recall_similar_flags("S1", ["duplicate"]) == []
    assert memory.reflect_on_supplier("S1") == ""


def test_hindsight_client_cache_is_thread_local(monkeypatch):
    import hindsight_client

    clients_ready = Barrier(2)

    class FakeHindsight:
        def __init__(self, **kwargs):
            self.thread_id = get_ident()
            clients_ready.wait(timeout=3)

    monkeypatch.setattr(hindsight_client, "Hindsight", FakeHindsight)
    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "MODE", "remote")
    monkeypatch.setattr(memory, "_client", None)
    monkeypatch.setattr(memory, "_client_failed", False)

    def get_clients(_):
        first = memory._get_client()
        second = memory._get_client()
        return first, second

    with ThreadPoolExecutor(max_workers=2) as executor:
        thread_clients = list(executor.map(get_clients, range(2)))

    assert thread_clients[0][0] is thread_clients[0][1]
    assert thread_clients[1][0] is thread_clients[1][1]
    assert thread_clients[0][0] is not thread_clients[1][0]
    assert thread_clients[0][0].thread_id != thread_clients[1][0].thread_id


def test_embedded_server_uses_groq_and_local_client(monkeypatch):
    calls = []
    module = ModuleType("hindsight")

    class FakeServerContext:
        def __init__(self, **kwargs):
            calls.append(("server-init", kwargs))

        def __enter__(self):
            calls.append(("server-start", None))
            return type("Server", (), {"url": "http://127.0.0.1:18888"})()

        def __exit__(self, *args):
            calls.append(("server-stop", args))

    class FakeClient:
        def __init__(self, base_url):
            self.base_url = base_url

    module.HindsightServer = FakeServerContext
    module.HindsightClient = FakeClient
    monkeypatch.setitem(sys.modules, "hindsight", module)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "MODE", "embedded")
    monkeypatch.setattr(memory, "LOCAL_PORT", 18888)
    monkeypatch.setattr(memory, "BASE_URL", "http://localhost:8888")
    monkeypatch.setattr(memory, "_client", None)
    monkeypatch.setattr(memory, "_client_failed", False)
    monkeypatch.setattr(memory, "_thread_clients", local())
    monkeypatch.setattr(memory, "_embedded_server_context", None)

    assert memory.start_embedded_server() is True
    client = memory._get_client()
    memory.stop_embedded_server()

    assert calls[0] == (
        "server-init",
        {
            "llm_provider": "groq",
            "llm_model": "openai/gpt-oss-20b",
            "llm_api_key": "test-groq-key",
            "host": "127.0.0.1",
            "port": 18888,
            "mcp_enabled": True,
        },
    )
    assert calls[-1][0] == "server-stop"
    assert client.base_url == "http://127.0.0.1:18888"


def test_fastapi_lifespan_starts_and_stops_embedded_hindsight(monkeypatch):
    api_module = importlib.import_module("api.api")
    events = []
    monkeypatch.setattr(memory, "MODE", "embedded")
    monkeypatch.setattr(memory, "start_embedded_server", lambda: events.append("start"))
    monkeypatch.setattr(memory, "stop_embedded_server", lambda: events.append("stop"))

    async def exercise_lifespan():
        async with api_module.app_lifespan(api_module.FastAPI()):
            assert events == ["start"]

    asyncio.run(exercise_lifespan())
    assert events == ["start", "stop"]


def test_retain_helpers_send_tag_supplier_and_decision_to_client(monkeypatch):
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def retain(self, **kwargs):
            self.calls.append(kwargs)

    client = RecordingClient()
    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "_client", client)
    monkeypatch.setattr(memory, "_client_failed", False)

    assert memory.retain_alert("S77", "Acme", "PO9", "HIGH", 82, [], "Hold PO")
    assert memory.retain_human_decision("S77", "Acme", "A9", "escalate", "Lee")
    assert memory.retain_fraud_review("S77", "Acme", "T9", ["duplicate"], "HIGH", "dismiss")
    assert memory.retain_action("S77", "Acme", "hold PO", "executed")
    assert memory.retain_outcome("S77", "Acme", "delay resolved", "hold PO")
    assert memory.retain_external_event("S77", "Acme", "recall", "Product recalled")

    contents = [call["content"] for call in client.calls]
    expected = [
        ("[alert_raised]", "S77", "Hold PO"),
        ("[human_decision]", "S77", "ESCALATE"),
        ("[fraud_flag_review]", "S77", "DISMISS"),
        ("[action_executed]", "S77", "hold PO"),
        ("[action_outcome]", "S77", "delay resolved"),
        ("[external_event]", "S77", "Product recalled"),
    ]
    for content, (tag, supplier_id, event_text) in zip(contents, expected):
        assert tag in content
        assert supplier_id in content
        assert event_text in content


def test_predict_output_is_independent_of_memory(monkeypatch):
    api_module = importlib.import_module("api.api")
    prediction = {
        "po_id": "PO-MEM-1",
        "risk_score": 73.2,
        "risk_level": "HIGH",
        "quality_rate": 91.0,
        "delivery_delay_rate": 18.0,
        "primary_risk_reason": "Supplier quality performance is below preferred range",
        "secondary_risk_reason": "Recent supplier performance is within normal ranges",
    }

    class FakePredictor:
        def predict_po_risk(self, po):
            return prediction.copy()

    class FakePipeline:
        def __init__(self):
            self.engine = object()

    monkeypatch.setattr(api_module.RiskPredictor, "load", lambda _: FakePredictor())
    monkeypatch.setattr(api_module.FraudEngine, "load", lambda _: object())
    monkeypatch.setattr(api_module, "RiskWisePipeline", FakePipeline)
    monkeypatch.setattr(memory, "recall_supplier_context", lambda *args, **kwargs: ["junk memory context"])
    monkeypatch.setattr(memory, "retain_alert", lambda *args: True)

    request = {
        "po_id": "PO-MEM-1", "po_date": "2026-09-26", "supplier_id": "S1",
        "buyer_id": "B1", "department": "Operations", "currency": "USD",
        "payment_terms": "Net 30", "delivery_window_days": 12, "product_id": "P1",
        "quantity": 10, "unit_price": 42.5,
    }
    monkeypatch.setattr(memory, "ENABLED", True)
    with TestClient(api_module.create_app()) as client:
        enabled_response = client.post("/predict", json=request)
    monkeypatch.setattr(memory, "ENABLED", False)
    with TestClient(api_module.create_app()) as client:
        disabled_response = client.post("/predict", json=request)

    assert enabled_response.status_code == disabled_response.status_code == 200
    assert enabled_response.json() == disabled_response.json() == prediction


def test_memory_endpoints_and_background_outcome(monkeypatch):
    api_module = importlib.import_module("api.api")
    retained = []

    class FakePipeline:
        def __init__(self):
            self.engine = object()

    class HealthyClient:
        def recall(self, **kwargs):
            return {"results": []}

    monkeypatch.setattr(api_module.RiskPredictor, "load", lambda _: object())
    monkeypatch.setattr(api_module.FraudEngine, "load", lambda _: object())
    monkeypatch.setattr(api_module, "RiskWisePipeline", FakePipeline)
    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "_get_client", lambda: HealthyClient())
    monkeypatch.setattr(memory, "recall_supplier_context", lambda supplier_id: [f"history for {supplier_id}"])
    monkeypatch.setattr(
        memory,
        "retain_outcome",
        lambda *args: retained.append(args),
    )

    with TestClient(api_module.create_app()) as client:
        monkeypatch.setattr(memory, "ENABLED", False)
        assert client.get("/health").json() == {"status": "ok", "memory": "disabled"}
        monkeypatch.setattr(memory, "ENABLED", True)
        assert client.get("/health").json() == {"status": "ok", "memory": "up"}
        assert client.get("/suppliers/S77/memory").json() == ["history for S77"]
        response = client.post(
            "/outcomes",
            json={"supplier_id": "S77", "outcome": "delay resolved", "related_action": "hold PO"},
        )

    assert response.status_code == 202
    assert response.json() == {"status": "queued"}
    assert retained == [("S77", None, "delay resolved", "hold PO")]


def test_scoring_and_training_modules_do_not_import_memory():
    package_dir = Path(__file__).parent
    violations = []
    for path in package_dir.glob("*.py"):
        if path.name == "memory.py" or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "risk_engine.memory" for alias in node.names):
                    violations.append(path.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "risk_engine.memory" or (
                    node.module == "risk_engine"
                    and any(alias.name == "memory" for alias in node.names)
                ):
                    violations.append(path.name)

    assert violations == []