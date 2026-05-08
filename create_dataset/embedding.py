"""
embedding.py

Builds (or incrementally updates) ChromaDB collections from resume_texts.json
or job_texts.json using OpenAI text-embedding-3-small (1536 dims, cosine space).

Requires OPENAI_API_KEY in a .env file at the project root.

Run:
    python embedding.py                                    # embed resumes
    python embedding.py --dataset jobs                     # embed job postings
    python embedding.py --dataset jobs --limit 5000        # embed a sample
    python embedding.py --query "data scientist"           # build then query
    python embedding.py --query-only "devops"              # query existing DB
    python embedding.py --reset                            # drop and rebuild
"""

import argparse
import json
import os
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import RateLimitError

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR  = PROJECT_ROOT / "chroma_db"
BATCH_SIZE  = 100
MAX_CHARS   = 6000
# ~95k tokens/batch at 1M TPM limit → 6s gap keeps us safely under
BATCH_SLEEP = 6

DATASETS = {
    "resumes": {
        "texts_path":      PROJECT_ROOT / "resume_texts.json",
        "collection_name": "resumes",
        "convert_hint":    "create_dataset/convert_pdfs.py",
        "display_fields":  ["category"],
    },
    "jobs": {
        "texts_path":      PROJECT_ROOT / "job_texts.json",
        "collection_name": "job_postings",
        "convert_hint":    "create_dataset/convert_jobs.py",
        "display_fields":  ["title", "company", "location", "experience_level"],
    },
}


def get_embedding_function() -> OpenAIEmbeddingFunction:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )


def get_collection(client: chromadb.PersistentClient, collection_name: str) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _add_with_retry(collection: chromadb.Collection, ids, documents, metadatas, max_retries: int = 5) -> None:
    for attempt in range(max_retries):
        try:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            return
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 60
            print(f"\n  Rate limit hit, waiting {wait}s...  (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


def build_vector_db(dataset: str, limit: int | None = None, reset: bool = False) -> chromadb.Collection:
    cfg = DATASETS[dataset]
    texts_path      = cfg["texts_path"]
    collection_name = cfg["collection_name"]

    if not texts_path.exists():
        raise FileNotFoundError(
            f"{texts_path} not found — run {cfg['convert_hint']} first."
        )

    print(f"Loading from {texts_path} ...")
    records = json.loads(texts_path.read_text())
    if limit:
        records = records[:limit]
    print(f"  {len(records)} records loaded")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  Dropped existing '{collection_name}' collection")
        except Exception:
            pass

    collection   = get_collection(client, collection_name)
    existing_ids = set(collection.get(include=[])["ids"])
    new_records  = [r for r in records if r["id"] not in existing_ids]
    print(f"  {len(existing_ids)} already embedded, {len(new_records)} to add")

    display_fields = cfg["display_fields"]

    for i in range(0, len(new_records), BATCH_SIZE):
        batch = new_records[i : i + BATCH_SIZE]
        _add_with_retry(
            collection,
            ids=[r["id"] for r in batch],
            documents=[r["text"][:MAX_CHARS] for r in batch],
            metadatas=[{k: r.get(k, "") for k in display_fields} for r in batch],
        )
        done = min(i + BATCH_SIZE, len(new_records))
        print(f"  Embedded {done}/{len(new_records)}", end="\r", flush=True)
        time.sleep(BATCH_SLEEP)

    if new_records:
        print()

    total = collection.count()
    print(f"Collection '{collection_name}': {total} documents  (model: text-embedding-3-small)")
    return collection


def query_collection(collection: chromadb.Collection, query: str, n: int = 5) -> None:
    print(f"\nQuery: {query!r}  (top {n} matches)")
    print("-" * 60)
    results = collection.query(query_texts=[query], n_results=n)
    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), 1):
        snippet  = doc[:200].replace("\n", " ")
        meta_str = "  |  ".join(f"{k}: {v}" for k, v in meta.items() if v)
        print(f"  #{rank}  distance={dist:.4f}  {meta_str}")
        print(f"       {snippet}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector collections.")
    parser.add_argument("--dataset",    choices=["resumes", "jobs"], default="resumes",
                        help="Which dataset to embed (default: resumes).")
    parser.add_argument("--query",      metavar="TEXT", help="Build DB then run a query.")
    parser.add_argument("--query-only", metavar="TEXT", help="Query an existing DB (skip build).")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Only embed the first N records (useful for testing).")
    parser.add_argument("--reset",      action="store_true",
                        help="Drop and rebuild the collection from scratch.")
    parser.add_argument("-n",           type=int, default=5,
                        help="Number of query results to return (default: 5).")
    args = parser.parse_args()

    collection_name = DATASETS[args.dataset]["collection_name"]

    if args.query_only:
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = get_collection(client, collection_name)
        query_collection(collection, args.query_only, args.n)
    else:
        collection = build_vector_db(args.dataset, limit=args.limit, reset=args.reset)
        if args.query:
            query_collection(collection, args.query, args.n)


if __name__ == "__main__":
    main()
