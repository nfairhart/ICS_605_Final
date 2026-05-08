"""
Convert sample_matches.jsonl -> finetune_data/train.jsonl + val.jsonl
in Gemma 4 chat format ready for SFTTrainer.
"""
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED = 42
TRAIN_SIZE = 9_000
MAX_RESUME = 2500
MAX_JOB = 2500
MAX_CHARS = 14_000  # ~4000 tokens at 3.5 chars/token; drops outliers that exceed 4096 budget
SRC = PROJECT_ROOT / "sample_matches.jsonl"
OUT = PROJECT_ROOT / "finetune_data"

INSTRUCTION = """\
Analyze the match between this resume and job posting.
Return a JSON object with these exact fields:
  rationale            (str, 2-3 sentence overall assessment)
  matching_strengths   (list[str], 2-4 items)
  skill_gaps           (list[str], 2-5 items)
  ats_keywords_missing (list[str], 3-8 items)
  resume_improvements  (list[str], 2-4 items)
  recommended_activities (list[str] or null)
  match_score          (int, 0-100)
  experience_level_fit (str: "under-qualified" | "well-matched" | "over-qualified")

RESUME:
{resume}

JOB POSTING:
{job}"""

RESPONSE_FIELDS = [
    "rationale",
    "matching_strengths",
    "skill_gaps",
    "ats_keywords_missing",
    "resume_improvements",
    "recommended_activities",
    "match_score",
    "experience_level_fit",
]

EXPERIENCE_LABEL_MAP = {
    # clearly negative — scores cluster around 35
    "under-qualified": "under-qualified",
    "poorly matched": "under-qualified",
    "poorly qualified": "under-qualified",
    "somewhat under-qualified": "under-qualified",
    # partial fit — scores cluster around 55, closer to well-matched (65) than under-qualified (35)
    "partial fit": "well-matched",
    "partially qualified": "well-matched",
    "partially-qualified": "well-matched",
    "partially matched": "well-matched",
    "partially-matched": "well-matched",
    "partially suitable": "well-matched",
    "partially matches": "well-matched",
    "moderate": "well-matched",
    # clearly positive
    "well-matched": "well-matched",
    "good fit": "well-matched",
    "over-qualified": "over-qualified",
}


def normalize_record(rec: dict) -> dict:
    raw_label = rec.get("experience_level_fit", "")
    normalized = EXPERIENCE_LABEL_MAP.get(raw_label)
    if normalized is None:
        return None  # drop rows with unknown labels
    return {**rec, "experience_level_fit": normalized}


def format_pair(rec: dict) -> str:
    instruction = INSTRUCTION.format(
        resume=rec["resume_text"][:MAX_RESUME].strip(),
        job=rec["job_text"][:MAX_JOB].strip(),
    )
    response = json.dumps(
        {k: rec.get(k) for k in RESPONSE_FIELDS},
        indent=2,
    )
    return (
        f"<start_of_turn>user\n{instruction}\n<end_of_turn>\n"
        f"<start_of_turn>model\n{response}\n<end_of_turn>"
    )


def main():
    records = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    records = [r for r in (normalize_record(rec) for rec in records) if r is not None]
    print(f"After label normalization: {len(records)} records")

    formatted = [(rec, format_pair(rec)) for rec in records]
    formatted = [(rec, text) for rec, text in formatted if len(text) <= MAX_CHARS]
    print(f"After length filter: {len(formatted)} records")

    random.seed(SEED)
    random.shuffle(formatted)
    train_pairs = formatted[:TRAIN_SIZE]
    val_pairs = formatted[TRAIN_SIZE:]

    OUT.mkdir(exist_ok=True)
    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        out_path = OUT / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for _, text in pairs:
                f.write(json.dumps({"text": text}) + "\n")
        print(f"Wrote {len(pairs)} examples -> {out_path}")


if __name__ == "__main__":
    main()
