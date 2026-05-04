"""
chroma_client.py

Singleton ChromaDB client and semantic job search helper.
"""

import os
import sys
import types
from pathlib import Path
from functools import lru_cache

# Must stub onnxruntime before chromadb import to avoid 60-120s hang on Apple Silicon
_onnx_stub = types.ModuleType("chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2")
_onnx_stub.ONNXMiniLM_L6_V2 = type("ONNXMiniLM_L6_V2", (), {})
sys.modules["chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"] = _onnx_stub

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# chroma_db lives at the project root, one level above this file
_CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
_EMBED_MODEL = "text-embedding-3-small"


def _get_embedding_function() -> OpenAIEmbeddingFunction:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set — required for semantic search")
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=_EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(_CHROMA_DIR))


@lru_cache(maxsize=2)
def _get_collection(name: str) -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def search_jobs(query: str, n: int = 10) -> list[dict]:
    """
    Semantic search over the job_postings ChromaDB collection.
    Returns a list of dicts with keys: id, title, company, location,
    experience_level, text, distance.
    """
    collection = _get_collection("job_postings")
    results = collection.query(
        query_texts=[query],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    jobs = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    ids = results["ids"][0]

    for doc, meta, dist, job_id in zip(docs, metas, dists, ids):
        jobs.append({
            "id": job_id,
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "location": meta.get("location", ""),
            "experience_level": meta.get("experience_level", ""),
            "text": doc,
            "distance": round(dist, 4),
        })
    return jobs
