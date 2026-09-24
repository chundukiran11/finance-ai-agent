from fastapi.testclient import TestClient

from finance_agent.api import app as app_module


def test_chat_endpoint(agent, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_agent", lambda: agent)
    client = TestClient(app_module.app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/chat", json={"message": "How much did I spend in total?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["refused"] is False
