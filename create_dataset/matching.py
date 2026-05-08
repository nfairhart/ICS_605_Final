"""
matching.py

For each resume in ChromaDB, queries the top-k most semantically similar job
postings, scores each (resume, job) pair with gpt-4.1-nano using structured
output, and appends results to matches.jsonl.

Resumable: pairs already present in the output file are skipped on re-run.

Run:
    python matching.py                              # all resumes, top-2 jobs each
    python matching.py --top-k 3                    # top-3 semantic jobs per resume
    python matching.py --sample-per-category 20     # 20 random resumes per category
    python matching.py --add-random-jobs 2          # +2 random (likely mismatched) jobs
    python matching.py --category FINANCE           # one category only
    python matching.py --output my_matches.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# chromadb 1.5.x eagerly imports onnxruntime (its default ONNX embedding
# function) at package init time.  On macOS/Apple Silicon this causes a 60-120s
# hang while CoreML/MPS backends initialize — even though this script always
# passes pre-computed OpenAI embeddings and never calls any chromadb EF.
# Stubbing the module before the import skips onnxruntime entirely.
_onnx_stub = types.ModuleType(
    "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"
)
_onnx_stub.ONNXMiniLM_L6_V2 = type("ONNXMiniLM_L6_V2", (), {})
sys.modules["chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"] = _onnx_stub

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CHROMA_DIR    = PROJECT_ROOT / "chroma_db"
OUTPUT_FILE   = PROJECT_ROOT / "matches.jsonl"
SCORING_MODEL = "gpt-4.1-nano"
EMBED_MODEL   = "text-embedding-3-small"

# Domain-specific keyword queries per resume category.
# Used by --add-category-jobs to find highly aligned jobs likely to score 75-100.
CATEGORY_QUERIES: dict[str, str] = {
    "ACCOUNTANT":            "certified public accountant CPA tax audit financial reporting GAAP bookkeeping",
    "ADVOCATE":              "attorney lawyer legal advocate litigation law firm counsel",
    "AGRICULTURE":           "agricultural farming crop production soil science agronomy farm manager",
    "APPAREL":               "fashion apparel clothing merchandising textile buyer retail fashion designer",
    "ARTS":                  "artist creative visual arts fine art illustration gallery curator",
    "AUTOMOBILE":            "automotive mechanic vehicle technician dealership auto repair service",
    "AVIATION":              "pilot aviation airline flight operations aircraft captain first officer",
    "BANKING":               "bank branch manager loan officer credit analyst retail banking teller",
    "BPO":                   "customer service call center BPO outsourcing technical support representative",
    "BUSINESS-DEVELOPMENT":  "business development partnerships revenue growth strategy deal sales manager",
    "CHEF":                  "chef cook culinary kitchen food preparation restaurant executive chef sous",
    "CONSTRUCTION":          "construction project manager contractor site supervisor building engineer",
    "CONSULTANT":            "management consultant strategy advisory business analyst McKinsey Deloitte",
    "DESIGNER":              "graphic designer UX UI product designer visual design Figma creative",
    "DIGITAL-MEDIA":         "digital marketing social media SEO content creator analytics influencer",
    "ENGINEERING":           "software engineer developer Python Java backend frontend full-stack programming",
    "FINANCE":               "finance investment portfolio analyst financial modeling CFA equity research",
    "FITNESS":               "personal trainer fitness coach gym wellness strength conditioning health",
    "HEALTHCARE":            "registered nurse RN physician healthcare clinical hospital patient care",
    "HR":                    "human resources HR recruiter talent acquisition HRBP people operations",
    "INFORMATION-TECHNOLOGY": "IT systems administrator network security cloud AWS DevOps infrastructure",
    "PUBLIC-RELATIONS":      "public relations PR communications media spokesperson press releases",
    "SALES":                 "sales account executive quota revenue B2B SaaS business development",
    "TEACHER":               "teacher educator curriculum instruction classroom K-12 school district",
}

# Characters sent to the scoring model per document.
# ~3 k chars ≈ ~750 tokens; keeps each call under ~2 k input tokens.
SCORE_CHARS = 3000
# Characters stored in the output JSONL (enough for fine-tuning prompts).
STORE_CHARS = 4000

SYSTEM_PROMPT = """\
You are an expert recruiter and career coach. Given a resume and a job posting,
evaluate how well the candidate fits the role and provide actionable guidance.

Respond with a JSON object. Generate the fields in this order — assess context
before assigning the final score and category:

- rationale: 2–3 sentences summarising the overall fit. Lead with the strongest
  signal (domain match, seniority, a critical gap) before elaborating.
- matching_strengths: 2–4 specific things the resume already demonstrates well
  for this role (skills, experience, credentials, soft skills).
- skill_gaps: 2–5 concrete qualifications or skills the job requires that are
  absent or understated in the resume.
- ats_keywords_missing: 3–8 important terms/phrases from the job description
  that do not appear in the resume and would help it pass automated screening.
- resume_improvements: 2–4 specific, actionable edits the candidate could make
  to their resume to better target this type of role (reword bullet points,
  quantify achievements, add a section, etc.).
- recommended_activities: Optional. Only include if there are meaningful gaps
  the candidate could close through real-world experience. Tailor to the field —
  this could be a side project, portfolio piece, open-source contribution,
  freelance work, internship, certification, workshop, volunteer role, community
  service, pro-bono work, professional association, competition, or any other
  activity that builds relevant credentials. Null if the resume is already
  well-rounded or the gaps are better addressed by resume framing alone.
- match_score: Integer 0–100 based on your assessment above.
    0–30  = poor fit   (missing key skills or wrong domain entirely)
   31–60 = partial fit (some relevant skills, could be considered)
   61–80 = good fit    (solid match, meets most requirements)
   81–100 = excellent  (strong match, exceeds most requirements)
  Be calibrated — use the full range. Most pairs should score 20–70.
- experience_level_fit: One of "under-qualified", "well-matched", or
  "over-qualified" based on seniority and scope relative to the role."""

USER_TEMPLATE = """\
## Resume
{resume_text}

## Job Posting
{job_text}

Evaluate this candidate's fit and provide coaching guidance."""


class MatchScore(BaseModel):
    rationale: str = Field(..., description="2–3 sentence overall assessment")
    matching_strengths: list[str] = Field(
        ..., description="2–4 resume strengths relevant to this role"
    )
    skill_gaps: list[str] = Field(
        ..., description="2–5 missing skills or qualifications"
    )
    ats_keywords_missing: list[str] = Field(
        ..., description="3–8 keywords from the JD absent in the resume"
    )
    resume_improvements: list[str] = Field(
        ..., description="2–4 specific resume edits for this role type"
    )
    recommended_activities: list[str] | None = Field(
        None,
        description=(
            "1–3 activities to close experience gaps — projects, volunteer work, "
            "certifications, community service, competitions, etc. Null if not needed."
        ),
    )
    match_score: int = Field(..., ge=0, le=100)
    experience_level_fit: str = Field(
        ..., description="One of: under-qualified, well-matched, over-qualified"
    )


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)


def get_embedding_function() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBED_MODEL,
    )


def batch_embed(texts: list[str], client: OpenAI, batch_size: int = 512) -> list[list[float]]:
    """Embed texts in batches. Far faster than one API call per query."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend(e.embedding for e in sorted_data)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts ...", end="\r", flush=True)
    print()
    return all_embeddings


def load_done_pairs(output_path: Path) -> set[tuple[str, str]]:
    """Return the set of (resume_id, job_id) already written to output."""
    if not output_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["resume_id"], rec["job_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def score_pair(
    client: OpenAI,
    resume_text: str,
    job_text: str,
    max_retries: int = 5,
) -> MatchScore:
    user_content = USER_TEMPLATE.format(
        resume_text=resume_text[:SCORE_CHARS],
        job_text=job_text[:SCORE_CHARS],
    )
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=SCORING_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format=MatchScore,
            )
            return response.choices[0].message.parsed
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = min(5 * (2 ** attempt), 60) + random.uniform(0, 3)
            print(f"\n  Rate limit — waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


def sample_resumes_by_category(
    resumes: list[dict], n: int, seed: int = 42
) -> list[dict]:
    """Return n randomly sampled resumes from each category."""
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in resumes:
        by_cat[r.get("category", "UNKNOWN")].append(r)
    rng = random.Random(seed)
    sampled = []
    for cat, cat_resumes in sorted(by_cat.items()):
        take = min(n, len(cat_resumes))
        sampled.extend(rng.sample(cat_resumes, take))
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Score resume–job pairs and write matches.jsonl."
    )
    parser.add_argument(
        "--top-k", type=int, default=2,
        help="Semantic jobs to retrieve per resume via ChromaDB (default: 2).",
    )
    parser.add_argument(
        "--sample-per-category", type=int, default=None, metavar="N",
        help="Randomly sample N resumes from each category instead of using all.",
    )
    parser.add_argument(
        "--add-random-jobs", type=int, default=0, metavar="K",
        help="Add K randomly drawn jobs per resume to inject low-score examples.",
    )
    parser.add_argument(
        "--add-category-jobs", type=int, default=0, metavar="K",
        help="Add K jobs retrieved using category-specific keywords (targets 75-100 scores).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total resumes after sampling (useful for a quick test).",
    )
    parser.add_argument(
        "--category", default=None,
        help="Filter resumes to a single category (e.g. FINANCE, HEALTHCARE).",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Concurrent OpenAI scoring threads (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help=f"Output JSONL path (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--max-total", type=int, default=None, metavar="N",
        help="Stop once the output file reaches N total pairs (existing + new).",
    )
    parser.add_argument(
        "--progress-file", type=Path, default=PROJECT_ROOT / "matching_progress.json",
        help="JSON file updated every 10 pairs with count/rate/ETA (default: matching_progress.json).",
    )
    args = parser.parse_args()

    # ── Load resumes ──────────────────────────────────────────────────────────
    print("Loading resumes ...")
    resumes: list[dict] = json.loads((PROJECT_ROOT / "resume_texts.json").read_text())
    if args.category:
        resumes = [
            r for r in resumes
            if r.get("category", "").upper() == args.category.upper()
        ]
        print(f"  Filtered to '{args.category}': {len(resumes)} resumes")
    if args.sample_per_category:
        resumes = sample_resumes_by_category(resumes, args.sample_per_category, args.seed)
        print(f"  Sampled {args.sample_per_category} per category → {len(resumes)} resumes")
    if args.limit:
        resumes = resumes[: args.limit]
    print(f"  Will process {len(resumes)} resumes")

    # ── Load job metadata for full-text lookup ────────────────────────────────
    print("Loading job metadata ...")
    all_jobs: list[dict] = json.loads((PROJECT_ROOT / "job_texts.json").read_text())
    jobs_by_id: dict[str, dict] = {j["id"]: j for j in all_jobs}

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    jobs_col = chroma.get_collection(
        name="job_postings",
        embedding_function=get_embedding_function(),
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_client = get_openai_client()

    # ── Resume from prior run ─────────────────────────────────────────────────
    done_pairs = load_done_pairs(args.output)
    print(f"  {len(done_pairs)} pairs already scored — will skip\n")

    rng = random.Random(args.seed)

    # ── Phase 1a: load resume embeddings from local ChromaDB (no API calls) ──────
    resumes_col = chroma.get_collection(
        name="resumes",
        embedding_function=get_embedding_function(),
    )
    print(f"Loading {len(resumes)} resume embeddings from local ChromaDB ...")
    resume_ids = [str(r["id"]) for r in resumes]
    stored = resumes_col.get(ids=resume_ids, include=["embeddings"])
    resume_emb_map: dict[str, list[float]] = dict(zip(stored["ids"], stored["embeddings"]))
    missing = [r for r in resumes if str(r["id"]) not in resume_emb_map]
    if missing:
        print(f"  {len(missing)} resumes not in ChromaDB — batch-embedding them now ...")
        new_vecs = batch_embed([r["text"][:SCORE_CHARS] for r in missing], openai_client)
        for r, vec in zip(missing, new_vecs):
            resume_emb_map[str(r["id"])] = vec
    print(f"  Loaded {len(resume_emb_map)} resume embeddings (0 API calls for stored resumes)")

    cat_embeddings: dict[str, list[float]] = {}
    if args.add_category_jobs > 0:
        unique_cats = sorted({r.get("category", "") for r in resumes} & CATEGORY_QUERIES.keys())
        print(f"Batch-embedding {len(unique_cats)} category keyword queries ...")
        cat_vecs = batch_embed([CATEGORY_QUERIES[c] for c in unique_cats], openai_client)
        cat_embeddings = dict(zip(unique_cats, cat_vecs))
        print(f"  Done (1 batch API call for {len(unique_cats)} categories)")

    # ── Phase 1b: build pair list using stored embeddings ────────────────────────
    print(f"Building pair list for {len(resumes)} resumes ...")
    pairs: list[dict] = []

    for r_idx, resume in enumerate(resumes, 1):
        resume_emb = resume_emb_map.get(str(resume["id"]))
        if resume_emb is None:
            continue
        results = jobs_col.query(
            query_embeddings=[resume_emb],
            n_results=args.top_k,
            include=["documents", "metadatas", "distances"],
        )
        semantic_ids = set(results["ids"][0])

        for job_id, job_doc, job_meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if (resume["id"], job_id) in done_pairs:
                continue
            job_full = jobs_by_id.get(job_id, {"id": job_id})
            pairs.append({
                "resume":    resume,
                "job_full":  job_full,
                "job_doc":   job_doc,
                "job_meta":  job_meta,
                "distance":  distance,
                "pair_type": "semantic",
            })

        print(f"  [{r_idx:>4}/{len(resumes)}]  pairs so far: {len(pairs)}", end="\r", flush=True)

        if args.add_random_jobs > 0:
            candidates = [
                j for j in all_jobs
                if j["id"] not in semantic_ids
                and (resume["id"], j["id"]) not in done_pairs
            ]
            for rand_job in rng.sample(candidates, min(args.add_random_jobs, len(candidates))):
                pairs.append({
                    "resume":    resume,
                    "job_full":  rand_job,
                    "job_doc":   rand_job.get("text", ""),
                    "job_meta":  {},
                    "distance":  None,
                    "pair_type": "random",
                })

        if args.add_category_jobs > 0:
            cat_emb = cat_embeddings.get(resume.get("category", ""))
            if cat_emb:
                cat_results = jobs_col.query(
                    query_embeddings=[cat_emb],
                    n_results=args.add_category_jobs + len(semantic_ids),
                    include=["documents", "metadatas", "distances"],
                )
                added = 0
                for job_id, job_doc, job_meta, distance in zip(
                    cat_results["ids"][0],
                    cat_results["documents"][0],
                    cat_results["metadatas"][0],
                    cat_results["distances"][0],
                ):
                    if added >= args.add_category_jobs:
                        break
                    if job_id in semantic_ids:
                        continue
                    if (resume["id"], job_id) in done_pairs:
                        continue
                    job_full = jobs_by_id.get(job_id, {"id": job_id})
                    pairs.append({
                        "resume":    resume,
                        "job_full":  job_full,
                        "job_doc":   job_doc,
                        "job_meta":  job_meta,
                        "distance":  distance,
                        "pair_type": "category",
                    })
                    semantic_ids.add(job_id)
                    added += 1

    print(f"\n  Done building pair list: {len(pairs)} pairs from {len(resumes)} resumes")

    if args.max_total:
        budget = args.max_total - len(done_pairs)
        if budget <= 0:
            print(f"Already at or above --max-total {args.max_total}. Nothing to do.")
            return
        rng.shuffle(pairs)
        pairs = pairs[:budget]
        print(f"  --max-total {args.max_total}: keeping {len(pairs)} pairs (budget remaining)")

    total = len(pairs)
    print(f"  {total} pairs to score  ({len(done_pairs)} already done)")
    print(f"\nScoring {total} pairs with {args.workers} workers  →  {args.output}\n")

    # ── Phase 2: score in parallel ────────────────────────────────────────────
    def score_one(pair: dict) -> dict | None:
        resume   = pair["resume"]
        job_full = pair["job_full"]
        job_text = job_full.get("text", pair["job_doc"])
        try:
            result = score_pair(openai_client, resume["text"], job_text)
        except Exception as exc:
            print(f"\n  ERROR ({resume['id']}, {job_full.get('id','')}): {exc}")
            return None

        job_meta = pair["job_meta"]
        return {
            "resume_id":              resume["id"],
            "resume_category":        resume.get("category", ""),
            "resume_text":            resume["text"][:STORE_CHARS],
            "job_id":                 job_full.get("id", ""),
            "job_title":              job_meta.get("title")    or job_full.get("title", ""),
            "job_company":            job_meta.get("company")  or job_full.get("company", ""),
            "job_location":           job_meta.get("location") or job_full.get("location", ""),
            "job_experience":         job_meta.get("experience_level") or job_full.get("experience_level", ""),
            "job_text":               job_text[:STORE_CHARS],
            "pair_type":              pair["pair_type"],
            "cosine_distance":        round(pair["distance"], 6) if pair["distance"] is not None else None,
            "match_score":            result.match_score,
            "experience_level_fit":   result.experience_level_fit,
            "rationale":              result.rationale,
            "matching_strengths":     result.matching_strengths,
            "skill_gaps":             result.skill_gaps,
            "ats_keywords_missing":   result.ats_keywords_missing,
            "resume_improvements":    result.resume_improvements,
            "recommended_activities": result.recommended_activities,
            "scored_at":              datetime.now(timezone.utc).isoformat(),
        }

    scored = 0
    errors = 0
    start_time = time.time()
    progress_lock = Lock()

    # Write initial progress file so monitoring works from the first second
    args.progress_file.write_text(json.dumps({
        "scored_this_run": 0,
        "errors": 0,
        "total_in_file": len(done_pairs),
        "target": args.max_total,
        "progress_pct": round(len(done_pairs) / args.max_total * 100, 1) if args.max_total else None,
        "pairs_per_min": 0,
        "eta_minutes": None,
        "status": "starting",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    def write_progress(i: int) -> None:
        elapsed = time.time() - start_time
        rate = scored / elapsed if elapsed > 0 else 0
        remaining = (total - i) / rate if rate > 0 else 0
        total_in_file = len(done_pairs) + scored
        data = {
            "scored_this_run": scored,
            "errors": errors,
            "total_in_file": total_in_file,
            "target": args.max_total,
            "progress_pct": round(total_in_file / args.max_total * 100, 1) if args.max_total else None,
            "pairs_per_min": round(rate * 60, 1),
            "eta_minutes": round(remaining / 60, 1),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            args.progress_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    with args.output.open("a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(score_one, p): p for p in pairs}
            for i, future in enumerate(as_completed(futures), 1):
                record = future.result()
                with progress_lock:
                    if record:
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        scored += 1
                    else:
                        errors += 1
                    total_so_far = len(done_pairs) + scored
                    target_str = f"/ {args.max_total}" if args.max_total else ""
                    print(
                        f"  [{i:>4}/{total}]  total: {total_so_far}{target_str}  "
                        f"scored: {scored}  errors: {errors}",
                        end="\r", flush=True,
                    )
                    if i % 10 == 0:
                        write_progress(i)

    write_progress(total)
    size_kb = args.output.stat().st_size // 1024 if args.output.exists() else 0
    print(f"\n\nDone.  {scored} new pairs scored,  {errors} errors.")
    print(f"Output: {args.output}  ({size_kb} KB)")
    if args.progress_file.exists():
        print(f"Progress log: {args.progress_file}")


if __name__ == "__main__":
    main()
