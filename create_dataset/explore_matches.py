"""
explore_matches.py  —  quick stats on matches.jsonl (or any matches file)

Run:
    python explore_matches.py
    python explore_matches.py --file test_matches.jsonl
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bucket(score: int) -> str:
    if score <= 30:  return "0–30  poor"
    if score <= 60:  return "31–60 partial"
    if score <= 80:  return "61–80 good"
    return                  "81–100 excellent"


def bar(value: float, total: float, width: int = 30) -> str:
    filled = round(value / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=PROJECT_ROOT / "test_matches.jsonl")
    parser.add_argument("--sample", type=int, default=2,
                        help="Number of full records to print at the end (default 2).")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}")
        return

    records = load(args.file)
    n = len(records)

    section(f"OVERVIEW  —  {args.file}  ({n} pairs)")
    scores        = [r["match_score"] for r in records]
    distances     = [r["cosine_distance"] for r in records]
    categories    = Counter(r["resume_category"] for r in records)
    unique_resumes = len({r["resume_id"] for r in records})
    unique_jobs    = len({r["job_id"] for r in records})

    print(f"  Pairs:           {n}")
    print(f"  Unique resumes:  {unique_resumes}")
    print(f"  Unique jobs:     {unique_jobs}")
    print(f"  Resume categories covered: {len(categories)}")

    section("SCORE DISTRIBUTION")
    print(f"  Mean:    {statistics.mean(scores):.1f}")
    print(f"  Median:  {statistics.median(scores):.1f}")
    print(f"  Stdev:   {statistics.stdev(scores):.1f}") if n > 1 else None
    print(f"  Min:     {min(scores)}   Max: {max(scores)}")
    print()
    buckets = Counter(bucket(s) for s in scores)
    for label in ["0–30  poor", "31–60 partial", "61–80 good", "81–100 excellent"]:
        count = buckets.get(label, 0)
        print(f"  {label:<18} {count:>4}  {bar(count, n)}  {count/n*100:.0f}%")

    section("EXPERIENCE LEVEL FIT")
    fits = Counter(r.get("experience_level_fit", "—") for r in records)
    for label, count in fits.most_common():
        print(f"  {label:<20} {count:>4}  {bar(count, n)}  {count/n*100:.0f}%")

    section("RESUME CATEGORIES")
    for cat, count in categories.most_common():
        print(f"  {cat:<30} {count:>4}  {bar(count, n)}")

    section("COSINE DISTANCE  (lower = more semantically similar)")
    print(f"  Mean:    {statistics.mean(distances):.4f}")
    print(f"  Median:  {statistics.median(distances):.4f}")
    print(f"  Min:     {min(distances):.4f}   Max: {max(distances):.4f}")

    # Correlation: does distance predict score?
    if n > 1:
        paired = sorted(records, key=lambda r: r["cosine_distance"])
        quartile = max(1, n // 4)
        closest = [r["match_score"] for r in paired[:quartile]]
        farthest = [r["match_score"] for r in paired[-quartile:]]
        print(f"\n  Avg score — closest {quartile} pairs:   {statistics.mean(closest):.1f}")
        print(f"  Avg score — farthest {quartile} pairs:  {statistics.mean(farthest):.1f}")

    section("RECOMMENDED ACTIVITIES  (null = resume already adequate)")
    has_activities = sum(1 for r in records if r.get("recommended_activities"))
    print(f"  With activities:  {has_activities} / {n}  ({has_activities/n*100:.0f}%)")
    print(f"  Null:             {n - has_activities} / {n}  ({(n-has_activities)/n*100:.0f}%)")

    if has_activities:
        print("\n  Sample activities suggested:")
        for r in records:
            acts = r.get("recommended_activities")
            if acts:
                print(f"    [{r['resume_category']}]  {acts[0]}")

    section("TOP ATS KEYWORDS MISSING  (most frequent across all pairs)")
    all_keywords: list[str] = []
    for r in records:
        all_keywords.extend(r.get("ats_keywords_missing", []))
    keyword_counts = Counter(k.lower().strip() for k in all_keywords)
    for kw, count in keyword_counts.most_common(15):
        print(f"  {kw:<40} {count:>3}x")

    section("SKILL GAPS  (most frequent)")
    all_gaps: list[str] = []
    for r in records:
        all_gaps.extend(r.get("skill_gaps", []))
    gap_counts = Counter(g.lower().strip()[:60] for g in all_gaps)
    for gap, count in gap_counts.most_common(10):
        print(f"  {gap:<60} {count:>3}x")

    if args.sample > 0:
        section(f"SAMPLE RECORDS  (first {args.sample})")
        for r in records[: args.sample]:
            print(f"\n  Resume:  [{r['resume_category']}]  id={r['resume_id']}")
            print(f"  Job:     {r['job_title']} @ {r['job_company']}")
            print(f"  Score:   {r['match_score']}  ({r.get('experience_level_fit', '—')})")
            print(f"  Dist:    {r['cosine_distance']:.4f}")
            print(f"  Why:     {r['rationale'][:140]}...")
            print(f"  Gaps:    {r.get('skill_gaps', [])}")
            print(f"  ATS missing: {r.get('ats_keywords_missing', [])}")


if __name__ == "__main__":
    main()
