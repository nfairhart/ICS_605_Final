#!/usr/bin/env python3
"""
Evaluate the fine-tuned Gemma 4 E2B model against the held-out validation set.

Success metric (from proposal): >80% agreement between model match_score
and ground-truth (GPT-5-mini) match_score, defined as within ±10 points.

Run with the model loaded in LM Studio:
  1. Open LM Studio, load your GGUF, start the local server (port 1234)
  2. python evaluate_model.py

Optional flags:
  --val   path to val.jsonl         (default: finetune_data/val.jsonl)
  --n     number of examples to run (default: all)
  --tol   match_score agreement tolerance (default: 10)
  --out   save per-example results  (default: eval_results.jsonl)
"""
import argparse
import json
import random
import re
from pathlib import Path

import json as _json
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _LocalClient:
    """Minimal OpenAI-compatible client using only stdlib — no pip install needed."""
    def __init__(self, base_url):
        self._url = base_url.rstrip("/") + "/chat/completions"

    def complete(self, messages, temperature=0.1, max_tokens=600):
        body = _json.dumps({
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()

client = _LocalClient("http://localhost:1234/v1")


# ── Parse args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--val",  default=str(PROJECT_ROOT / "finetune_data/val.jsonl"))
parser.add_argument("--n",    type=int, default=None)
parser.add_argument("--seed", type=int, default=42,
                    help="random seed for sampling (default: 42)")
parser.add_argument("--tol",  type=int, default=10,
                    help="match_score ±tolerance for agreement (default 10)")
parser.add_argument("--out",     default=str(PROJECT_ROOT / "training/eval_results.jsonl"))
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--peek",    type=int, default=0,
                    help="print this many raw model outputs for inspection (default: 0)")
args = parser.parse_args()

WORKERS = args.workers

client = _LocalClient("http://localhost:1234/v1")

VALID_LABELS = {"under-qualified", "well-matched", "over-qualified"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_example(row: dict) -> tuple[str, dict]:
    """Split a val.jsonl row into (user_prompt, ground_truth_dict)."""
    text = row["text"]
    user_part  = text.split("<start_of_turn>user\n")[1].split("\n<end_of_turn>")[0]
    model_part = text.split("<start_of_turn>model\n")[1].split("\n<end_of_turn>")[0]
    return user_part.strip(), json.loads(model_part.strip())


def call_model(prompt: str) -> str:
    return client.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # greedy — deterministic and reproducible for eval
        max_tokens=4096,  # base model uses reasoning tokens before final output; 4096 covers thinking + response
    )


def extract_json(raw: str) -> dict | None:
    """Pull the first {...} block out of the model response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def score_agreement(pred_score: int | float, gt_score: int | float, tol: int) -> bool:
    return abs(pred_score - gt_score) <= tol


# ── Load val set ──────────────────────────────────────────────────────────────

val_path = Path(args.val)
rows = [json.loads(l) for l in val_path.read_text().splitlines() if l.strip()]
if args.n:
    random.seed(args.seed)
    rows = random.sample(rows, min(args.n, len(rows)))

print(f"Evaluating {len(rows)} examples from {val_path}")
print(f"Match-score agreement tolerance: ±{args.tol} points\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

completed  = 0
counter_lock = threading.Lock()

def evaluate_row(i: int, row: dict) -> dict:
    prompt, gt = parse_example(row)
    raw  = call_model(prompt)
    pred = extract_json(raw)

    result = {
        "i": i,
        "gt_score": gt.get("match_score"),
        "gt_label": gt.get("experience_level_fit"),
        "raw": raw,
        "pred": pred,
        "json_ok": pred is not None,
        "score_agreed": None,
        "label_correct": None,
    }

    REQUIRED_FIELDS = {
        "match_score", "experience_level_fit", "rationale",
        "matching_strengths", "skill_gaps", "ats_keywords_missing",
        "resume_improvements", "recommended_activities",
    }

    if pred is not None:
        missing = REQUIRED_FIELDS - pred.keys()
        result["schema_complete"] = len(missing) == 0
        result["missing_fields"]  = sorted(missing)

        try:
            ps = int(pred.get("match_score", -1))
            gs = int(gt["match_score"])
            err = abs(ps - gs)
            result["pred_score"] = ps
            result["score_error"] = err
            result["score_agreed"] = err <= args.tol
        except (TypeError, ValueError):
            pass

        pl = pred.get("experience_level_fit", "").strip().lower()
        gl = gt.get("experience_level_fit", "").strip().lower()
        if gl in VALID_LABELS:
            result["pred_label"] = pl
            result["label_correct"] = pl == gl

    return result


results_map = {}
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(evaluate_row, i, row): i for i, row in enumerate(rows, 1)}
    print(f"Submitted {len(futures)} tasks across {WORKERS} workers — waiting for first results...", flush=True)
    running_errors  = []
    running_agreed  = 0
    running_json_ok = 0
    running_label_correct = 0
    running_label_known   = 0

    for future in as_completed(futures):
        result = future.result()
        results_map[result["i"]] = result
        with counter_lock:
            completed += 1
            done = completed

        if result["json_ok"]:
            running_json_ok += 1
        if "score_error" in result:
            running_errors.append(result["score_error"])
            running_agreed += int(result.get("score_agreed", False))
        if "label_correct" in result:
            running_label_known   += 1
            running_label_correct += int(result["label_correct"])

        mae         = sum(running_errors) / max(len(running_errors), 1)
        score_pct   = 100 * running_agreed / max(len(running_errors), 1)
        label_pct   = 100 * running_label_correct / max(running_label_known, 1)
        print(f"[{done:4d}/{len(rows)}]  "
              f"score_agreement={score_pct:.1f}%  "
              f"label_acc={label_pct:.1f}%  "
              f"mae={mae:.1f}",
              flush=True)

        if args.peek > 0 and done <= args.peek:
            print(f"\n--- peek {done} ---")
            print(f"  GT : score={result['gt_score']}  label={result['gt_label']}")
            if result.get("pred"):
                print(f"  PRD: score={result.get('pred_score','?')}  label={result.get('pred_label','?')}")
            else:
                print(f"  RAW: {result['raw'][:300]}")
            print()

# restore original order
results      = [results_map[i] for i in range(1, len(rows) + 1)]

# aggregate
n_json_ok    = sum(1 for r in results if r["json_ok"])
n_schema_ok  = sum(1 for r in results if r.get("schema_complete"))
score_errors  = [r["score_error"] for r in results if "score_error" in r]
score_agreed = sum(1 for r in results if r.get("score_agreed"))
label_results = [r["label_correct"] for r in results if "label_correct" in r]
label_correct = sum(label_results)
label_known   = len(label_results)

print()  # newline after \r progress

# ── Summary ───────────────────────────────────────────────────────────────────

n = len(rows)
mae = sum(score_errors) / max(len(score_errors), 1)
score_agree_pct = 100 * score_agreed / max(len(score_errors), 1)
label_acc_pct   = 100 * label_correct / max(label_known, 1)
json_pct        = 100 * n_json_ok / n

print("\n" + "="*55)
print("EVALUATION SUMMARY")
print("="*55)
print(f"Examples evaluated      : {n}")
print(f"JSON parse rate         : {json_pct:.1f}%  ({n_json_ok}/{n})")
schema_pct = 100 * n_schema_ok / max(n_json_ok, 1)
print(f"Schema complete (8 fields): {schema_pct:.1f}%  ({n_schema_ok}/{n_json_ok} valid JSON rows)")
print(f"match_score MAE         : {mae:.1f} points")
print(f"match_score agreement   : {score_agree_pct:.1f}%  "
      f"(±{args.tol} pts)  [{score_agreed}/{len(score_errors)}]")
print(f"experience_level_fit acc: {label_acc_pct:.1f}%  "
      f"[{label_correct}/{label_known}]")
print()

# proposal target check
target = 80.0
status = "PASS ✓" if score_agree_pct >= target else "FAIL ✗"
print(f"Proposal target (>80% score agreement): {status}")
print("="*55)

# ── Save per-example results ───────────────────────────────────────────────────

out_path = Path(args.out)
with out_path.open("w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\nPer-example results saved to {out_path}")
