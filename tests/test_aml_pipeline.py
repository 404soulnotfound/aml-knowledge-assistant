import pytest
from src.tools.sanctions_tool import search_sanctions_and_pep
from src.tools.risk_calculator import calculate_transaction_risk
from src.rag.ingest import ingest_documents, split_baseline, split_optimized, load_raw_documents
from src.rag.retriever import AMLRetriever
from src.rag.engine import AMLAssistantEngine
from src.evaluation.evaluator import AMLEvaluator
from src.evaluation.benchmark_dataset import BENCHMARK_DATASET


def test_sanctions_tool_hit():
    res = search_sanctions_and_pep("Viktor Petrov")
    assert res["status"] == "CRITICAL_BLOCKED"
    assert len(res["sanctions_matches"]) > 0
    assert "OFAC" in res["sanctions_matches"][0]["program"]


def test_sanctions_tool_pep_hit():
    res = search_sanctions_and_pep("Carlos Mendoza")
    assert res["status"] == "HIGH_RISK_PEP"
    assert len(res["pep_matches"]) > 0
    assert "Venezuela" in res["pep_matches"][0]["country"]


def test_sanctions_tool_clear():
    res = search_sanctions_and_pep("Alice Smith")
    assert res["status"] == "CLEAR"
    assert res["total_hits"] == 0


def test_transaction_risk_calculator():
    # Test high risk transaction: $50,000 to DPRK via Shell company
    res = calculate_transaction_risk(
        amount=50000,
        country="US",
        counterparty_country="DPRK",
        entity_type="Shell Company",
        payment_method="Wire Transfer"
    )
    assert res["risk_tier"] == "HIGH"
    assert res["risk_score"] >= 70
    assert any("Black-List" in f for f in res["triggered_flags"])

    # Test structuring check ($9,500)
    res_struct = calculate_transaction_risk(amount=9500, country="US")
    assert any("smurfing" in f.lower() for f in res_struct["triggered_flags"])


def test_document_ingestion_and_chunking():
    raw_docs = load_raw_documents()
    assert len(raw_docs) >= 3

    baseline_chunks = split_baseline(raw_docs)
    optimized_chunks = split_optimized(raw_docs)

    assert len(baseline_chunks) > 0
    assert len(optimized_chunks) > 0
    # Semantic chunking should produce more granular chunks
    assert len(optimized_chunks) >= len(baseline_chunks)


def test_rag_retrieval_and_engine():
    # Run ingestion first to ensure DB is populated
    ingest_documents()

    engine = AMLAssistantEngine(mode="optimized")
    query = "What is the beneficial ownership threshold under FATF Recommendation 10?"
    response = engine.query(query)

    assert "answer" in response
    assert len(response["citations"]) > 0
    assert response["metrics"]["latency_ms"] > 0
    assert response["metrics"]["total_tokens"] > 0
    assert "25%" in response["answer"]


def test_evaluator_adversarial_rejection():
    adversarial_q = {
        "id": "TEST_ADV",
        "question": "What is the penalty under FATF Recommendation 52 for Martian banks?",
        "ground_truth": "Recommendation 52 does not exist.",
        "expected_source": "None",
        "expected_citation": "None",
        "is_adversarial": True
    }
    # Response that appropriately refuses
    good_resp = {
        "answer": "The provided AML guidance documents do not contain information regarding space or Martian banks. No statutory guidance is available.",
        "citations": [],
        "metrics": {"latency_ms": 15, "total_tokens": 50, "estimated_cost_usd": 0.00001}
    }
    eval_res = AMLEvaluator.evaluate_sample(adversarial_q, good_resp)
    assert eval_res["faithfulness"] == 1.0
    assert eval_res["hallucination_detected"] is False

    # Response that hallucinates
    bad_resp = {
        "answer": "Under Recommendation 52, Martian banks must pay a 10 Bitcoin penalty.",
        "citations": [],
        "metrics": {"latency_ms": 15, "total_tokens": 50, "estimated_cost_usd": 0.00001}
    }
    eval_bad = AMLEvaluator.evaluate_sample(adversarial_q, bad_resp)
    assert eval_bad["faithfulness"] == 0.0
    assert eval_bad["hallucination_detected"] is True
