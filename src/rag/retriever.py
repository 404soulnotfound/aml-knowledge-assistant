from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from ..config import settings
from .ingest import get_vector_store


class AMLRetriever:
    """
    Retrieves regulatory chunks from ChromaDB with relevance filtering,
    source metadata attribution, and support for baseline vs optimized collections.
    """
    def __init__(self, collection_name: str = "aml_optimized"):
        self.collection_name = collection_name
        self.collection = get_vector_store(collection_name)

    def retrieve(self, query: str, top_k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        k = top_k or settings.TOP_K
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_dict
        )
        
        retrieved_items: List[Dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return retrieved_items

        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") and results["distances"] else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

        for doc_text, meta, dist, chunk_id in zip(docs, metadatas, distances, ids):
            # Chroma cosine distance is in [0, 2]; map to [0, 1] similarity
            d_val = float(dist) if dist is not None else 0.0
            similarity = round(max(0.0, 1.0 - (d_val / 2.0)), 4)
            retrieved_items.append({
                "chunk_id": chunk_id,
                "content": doc_text,
                "metadata": meta,
                "similarity_score": similarity,
                "source": meta.get("source", "Unknown"),
                "section": meta.get("section", "General")
            })

        return retrieved_items
