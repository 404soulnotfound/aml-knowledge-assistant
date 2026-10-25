import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import router as api_router
from .middleware import RequestLatencyMiddleware
from ..rag.ingest import ingest_documents, get_chroma_client
from ..config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure vector collections are initialized at startup
    try:
        client = get_chroma_client()
        cols = [c.name for c in client.list_collections()]
        if "aml_optimized" not in cols or "aml_baseline" not in cols:
            print("[Lifespan] Initializing and indexing documents into ChromaDB...")
            ingest_documents()
    except Exception as e:
        print(f"[Lifespan Warning] Ingestion on startup error: {e}")
    yield


app = FastAPI(
    title="AML Knowledge Assistant — RAG + LLM Evaluation",
    description="Regulatory guidance assistant over FATF, FinCEN & Wolfsberg standards with Sanctions/Risk tool-calling, LLM evaluation benchmark, and cost/latency tracking.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Latency Header Middleware
app.add_middleware(RequestLatencyMiddleware)

# API routes
app.include_router(api_router, prefix="/api")

# Static files & Web UI
static_dir = settings.BASE_DIR / "src" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def serve_ui():
        return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
