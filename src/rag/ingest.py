import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings


class GeminiOrLocalEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function that uses Google Generative AI embeddings when an API key is present,
    and falls back to deterministic high-dimensional embeddings for offline development/testing.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key or settings.effective_api_key
        self.model_name = model_name
        self._gemini_client = None

    def name(self) -> str:
        return "gemini_or_local"
        
        if self.api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._gemini_client = GoogleGenerativeAIEmbeddings(
                    model=self.model_name,
                    google_api_key=self.api_key
                )
            except Exception as e:
                print(f"[EmbeddingFunction] Gemini embeddings init failed ({e}), falling back to offline mode.")

    def __call__(self, input: Documents) -> Embeddings:
        if self._gemini_client:
            try:
                return self._gemini_client.embed_documents(input)
            except Exception as e:
                print(f"[EmbeddingFunction] Gemini API call error: {e}. Falling back to deterministic vectors.")
                
        # Deterministic 768-dimensional fallback embedding based on token n-grams and hashing
        return [self._offline_embed(doc) for doc in input]

    def _offline_embed(self, text: str, dim: int = 768) -> List[float]:
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec.tolist()
            
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * (1.0 + 1.0 / (i + 1))
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def load_raw_documents() -> List[Document]:
    """Loads all markdown policy documents from the documents directory."""
    docs: List[Document] = []
    doc_dir = settings.DOCUMENTS_DIR
    if not doc_dir.exists():
        return docs

    for file_path in doc_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            title = file_path.stem.replace("_", " ").title()
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "title": title,
                    "path": str(file_path)
                }
            ))
    return docs


def split_baseline(docs: List[Document]) -> List[Document]:
    """
    Naive recursive character splitting without header or semantic structure preservation.
    Used as the Baseline benchmark.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.BASELINE_CHUNK_SIZE,
        chunk_overlap=settings.BASELINE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"baseline_{i}"
        c.metadata["strategy"] = "baseline"
    return chunks


def split_optimized(docs: List[Document]) -> List[Document]:
    """
    Optimized semantic splitting:
    1. Splits across Markdown sections (## headings)
    2. Prepends section hierarchy metadata to every chunk
    3. Finer chunk sizes (500 tokens) with tighter overlap for high precision retrieval.
    """
    optimized_chunks: List[Document] = []
    chunk_counter = 0

    for doc in docs:
        sections = re.split(r"(^##\s+.+$)", doc.page_content, flags=re.MULTILINE)
        doc_header = sections[0].strip() if sections else ""
        current_section_title = "General"

        for idx, part in enumerate(sections[1:]):
            part = part.strip()
            if not part:
                continue
            if part.startswith("## "):
                current_section_title = part.replace("## ", "").strip()
                continue
            
            # Sub-split long sections with overlap
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.OPTIMIZED_CHUNK_SIZE,
                chunk_overlap=settings.OPTIMIZED_CHUNK_OVERLAP,
                separators=["\n### ", "\n\n", "\n- ", "\n", " "]
            )
            sub_chunks = sub_splitter.split_text(part)
            
            for sub_text in sub_chunks:
                # Prepend contextual regulatory header
                enriched_text = f"[{doc.metadata.get('title')}] > [{current_section_title}]\n{sub_text}"
                optimized_chunks.append(Document(
                    page_content=enriched_text,
                    metadata={
                        "source": doc.metadata.get("source"),
                        "title": doc.metadata.get("title"),
                        "section": current_section_title,
                        "chunk_id": f"opt_{chunk_counter}",
                        "strategy": "optimized"
                    }
                ))
                chunk_counter += 1

    return optimized_chunks


def get_chroma_client() -> chromadb.PersistentClient:
    persist_dir = str(settings.CHROMA_PERSIST_DIR)
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_vector_store(collection_name: str = "aml_optimized"):
    client = get_chroma_client()
    embedding_fn = GeminiOrLocalEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_documents() -> Dict[str, int]:
    """
    Runs ingestion for both 'aml_baseline' and 'aml_optimized' collections.
    Returns chunk counts for both strategies.
    """
    raw_docs = load_raw_documents()
    if not raw_docs:
        print("[Ingestion] No documents found in data/documents.")
        return {"baseline": 0, "optimized": 0}

    client = get_chroma_client()
    embedding_fn = GeminiOrLocalEmbeddingFunction()

    # Ingest Baseline
    baseline_chunks = split_baseline(raw_docs)
    baseline_col = client.get_or_create_collection("aml_baseline", embedding_function=embedding_fn)
    # Clear existing
    existing_ids = baseline_col.get()["ids"]
    if existing_ids:
        baseline_col.delete(ids=existing_ids)
    
    baseline_col.add(
        ids=[c.metadata["chunk_id"] for c in baseline_chunks],
        documents=[c.page_content for c in baseline_chunks],
        metadatas=[c.metadata for c in baseline_chunks]
    )

    # Ingest Optimized
    optimized_chunks = split_optimized(raw_docs)
    optimized_col = client.get_or_create_collection("aml_optimized", embedding_function=embedding_fn)
    existing_opt_ids = optimized_col.get()["ids"]
    if existing_opt_ids:
        optimized_col.delete(ids=existing_opt_ids)

    optimized_col.add(
        ids=[c.metadata["chunk_id"] for c in optimized_chunks],
        documents=[c.page_content for c in optimized_chunks],
        metadatas=[c.metadata for c in optimized_chunks]
    )

    print(f"[Ingestion] Successfully indexed {len(baseline_chunks)} baseline chunks and {len(optimized_chunks)} optimized chunks.")
    return {
        "baseline": len(baseline_chunks),
        "optimized": len(optimized_chunks)
    }


if __name__ == "__main__":
    ingest_documents()
