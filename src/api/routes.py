from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..rag.engine import AMLAssistantEngine
from ..tools.sanctions_tool import search_sanctions_and_pep
from ..tools.risk_calculator import calculate_transaction_risk
from ..evaluation.run_experiment import run_full_benchmark
from ..evaluation.benchmark_dataset import BENCHMARK_DATASET
from ..rag.ingest import ingest_documents, load_raw_documents
from .middleware import metrics_tracker
from ..config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query or question regarding AML guidance or entities")
    mode: str = Field(default="optimized", description="'optimized' (semantic + anti-hallucination) or 'baseline'")
    api_key: Optional[str] = Field(default=None, description="Optional Gemini API key override")


class ScreenRequest(BaseModel):
    name: str = Field(..., description="Individual or Entity name to screen against OFAC/UN/EU watchlists & PEPs")
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class RiskScoreRequest(BaseModel):
    amount: float = Field(..., ge=0, description="Transaction amount")
    currency: str = Field(default="USD")
    country: str = Field(default="US")
    entity_type: str = Field(default="Individual")
    payment_method: str = Field(default="Wire Transfer")
    counterparty_country: str = Field(default="")


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    RAG Assistant query endpoint:
    Retrieves policy guidance from ChromaDB, executes tools (sanctions/risk) if triggered,
    applies anti-hallucination prompt, and tracks latency/token costs.
    """
    try:
        engine = AMLAssistantEngine(mode=req.mode, api_key=req.api_key)
        result = engine.query(req.query)
        
        # Record into metrics tracker
        metrics_tracker.record_chat(
            latency_ms=result["metrics"]["latency_ms"],
            in_tokens=result["metrics"]["input_tokens"],
            out_tokens=result["metrics"]["output_tokens"],
            cost_usd=result["metrics"]["estimated_cost_usd"]
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screen")
async def screen_endpoint(req: ScreenRequest):
    """
    Direct sanctions & Politically Exposed Persons (PEP) watchlist screening tool.
    """
    try:
        return search_sanctions_and_pep(name=req.name, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-score")
async def risk_score_endpoint(req: RiskScoreRequest):
    """
    Direct AML Transaction Risk Assessment & Statutory Obligations engine.
    """
    try:
        return calculate_transaction_risk(
            amount=req.amount,
            currency=req.currency,
            country=req.country,
            entity_type=req.entity_type,
            payment_method=req.payment_method,
            counterparty_country=req.counterparty_country
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """
    Returns live latency (p50, p95), cumulative token count, and estimated cost in USD.
    """
    return metrics_tracker.get_summary()


@router.post("/evaluate")
async def trigger_evaluation():
    """
    Runs automated benchmark experiment over 25 questions, comparing Baseline vs Optimized RAG.
    """
    try:
        return run_full_benchmark()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def reingest():
    """
    Re-indexes policy documents into ChromaDB.
    """
    try:
        return ingest_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """
    Lists indexed guidance documents and benchmark questions count.
    """
    raw = load_raw_documents()
    return {
        "document_count": len(raw),
        "documents": [
            {
                "title": d.metadata.get("title"),
                "source": d.metadata.get("source"),
                "characters": len(d.page_content)
            }
            for d in raw
        ],
        "benchmark_questions": len(BENCHMARK_DATASET)
    }
