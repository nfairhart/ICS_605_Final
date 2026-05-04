"""
test_connections.py

Quick diagnostic: checks ChromaDB collections and OpenAI API health.
Run: python test_connections.py
"""

import json
import os
import sys
import time
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Same onnxruntime stub used in matching.py
_onnx_stub = types.ModuleType(
    "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"
)
_onnx_stub.ONNXMiniLM_L6_V2 = type("ONNXMiniLM_L6_V2", (), {})
sys.modules["chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"] = _onnx_stub

import chromadb
from openai import OpenAI, RateLimitError, AuthenticationError

CHROMA_DIR = Path("chroma_db")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4.1-nano"

ok = True

def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {label}" + (f"  —  {detail}" if detail else ""))
    if not passed:
        ok = False


# ── 1. ChromaDB ──────────────────────────────────────────────────────────────
print("\n=== ChromaDB ===")
try:
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    cols = {c.name: c for c in chroma.list_collections()}
    check("chroma_db directory found", CHROMA_DIR.exists())
    check("'job_postings' collection exists", "job_postings" in cols,
          f"found: {list(cols)}")
    check("'resumes' collection exists", "resumes" in cols)
    if "job_postings" in cols:
        n_jobs = cols["job_postings"].count()
        check("job_postings has documents", n_jobs > 0, f"{n_jobs} docs")
    if "resumes" in cols:
        n_res = cols["resumes"].count()
        check("resumes has documents", n_res > 0, f"{n_res} docs")
except Exception as e:
    check("ChromaDB init", False, str(e))

# ── 2. OpenAI API key ────────────────────────────────────────────────────────
print("\n=== OpenAI API ===")
api_key = os.getenv("OPENAI_API_KEY")
check("OPENAI_API_KEY set in .env", bool(api_key),
      "(not set)" if not api_key else f"...{api_key[-6:]}")

if api_key:
    client = OpenAI(api_key=api_key)

    # 2a. Embedding
    try:
        t0 = time.time()
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=["software engineer with Python experience"],
        )
        elapsed = time.time() - t0
        check(f"Embedding call ({EMBED_MODEL})", True,
              f"{len(resp.data[0].embedding)}-dim vector in {elapsed:.2f}s")
    except RateLimitError as e:
        check(f"Embedding call ({EMBED_MODEL})", False,
              f"RATE LIMITED — {e}")
    except AuthenticationError as e:
        check(f"Embedding call ({EMBED_MODEL})", False,
              f"AUTH ERROR — check your API key: {e}")
    except Exception as e:
        check(f"Embedding call ({EMBED_MODEL})", False, str(e))

    # 2b. Chat completion (tiny)
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
        )
        elapsed = time.time() - t0
        reply = resp.choices[0].message.content.strip()
        check(f"Chat completion ({CHAT_MODEL})", True,
              f"reply='{reply}'  in {elapsed:.2f}s")
    except RateLimitError as e:
        check(f"Chat completion ({CHAT_MODEL})", False,
              f"RATE LIMITED — {e}")
    except AuthenticationError as e:
        check(f"Chat completion ({CHAT_MODEL})", False,
              f"AUTH ERROR — {e}")
    except Exception as e:
        check(f"Chat completion ({CHAT_MODEL})", False, str(e))

    # 2c. Rate limit headers (usage tier info)
    try:
        resp = client.embeddings.with_raw_response.create(
            model=EMBED_MODEL,
            input=["test"],
        )
        rl_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower().startswith("x-ratelimit")
        }
        if rl_headers:
            print("\n  Rate-limit headers:")
            for k, v in sorted(rl_headers.items()):
                print(f"    {k}: {v}")
        else:
            print("  (no x-ratelimit headers returned)")
    except Exception:
        pass

# ── 3. Local data files ──────────────────────────────────────────────────────
print("\n=== Data files ===")
for fname in ("resume_texts.json", "job_texts.json"):
    p = Path(fname)
    if p.exists():
        data = json.loads(p.read_text())
        check(fname, True, f"{len(data)} records")
    else:
        check(fname, False, "file not found")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'All checks passed.' if ok else 'One or more checks FAILED — see above.'}\n")
sys.exit(0 if ok else 1)
