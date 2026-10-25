from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_documents():
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert data["document_count"] >= 3
    assert data["benchmark_questions"] == 25


def test_api_screen():
    res = client.post("/api/screen", json={"name": "Viktor Petrov"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "CRITICAL_BLOCKED"


def test_api_risk_score():
    res = client.post("/api/risk-score", json={
        "amount": 25000,
        "country": "US",
        "counterparty_country": "Venezuela",
        "entity_type": "Shell Corporation"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tier"] == "HIGH"
    assert data["risk_score"] >= 70


def test_api_chat():
    res = client.post("/api/chat", json={
        "query": "What is the beneficial ownership threshold under FATF?",
        "mode": "optimized"
    })
    assert res.status_code == 200
    data = res.json()
    assert "25%" in data["answer"]
    assert len(data["citations"]) > 0
    assert "metrics" in data
    assert data["metrics"]["total_tokens"] > 0


def test_api_metrics():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["total_requests"] >= 1
    assert "pricing_tier" in data["financial"]
