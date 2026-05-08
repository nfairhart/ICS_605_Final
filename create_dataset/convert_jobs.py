"""
convert_jobs.py

Reads linkedin-job-postings/postings.csv and writes job_texts.json.
Each record combines title, company, location, experience level, and description
into a single text field for embedding.

Run:
    python convert_jobs.py             # all 123k postings
    python convert_jobs.py --limit 10000  # smaller sample for testing
"""

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "linkedin-job-postings/postings.csv"
OUTPUT   = PROJECT_ROOT / "job_texts.json"


def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_job_text(row: dict) -> str:
    parts = []

    header = row["title"].strip()
    if row["company_name"].strip():
        header += f" at {row['company_name'].strip()}"
    if header:
        parts.append(header)

    details = []
    if row["location"].strip():
        details.append(f"Location: {row['location'].strip()}")
    if row["formatted_experience_level"].strip():
        details.append(f"Level: {row['formatted_experience_level'].strip()}")
    if row["formatted_work_type"].strip():
        details.append(f"Type: {row['formatted_work_type'].strip()}")
    if details:
        parts.append("  |  ".join(details))

    if row["description"].strip():
        parts.append(row["description"].strip())

    return clean_text("\n\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Convert LinkedIn job postings CSV to JSON.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of records (default: all).")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}\nExtract linkedin-job-postings.zip first.")

    print(f"Reading {CSV_PATH} ...")
    records = []
    skipped = 0

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if args.limit and len(records) >= args.limit:
                break
            text = build_job_text(row)
            if not text:
                skipped += 1
                continue
            records.append({
                "id":               row["job_id"],
                "text":             text,
                "title":            row["title"].strip(),
                "company":          row["company_name"].strip(),
                "location":         row["location"].strip(),
                "experience_level": row["formatted_experience_level"].strip(),
                "work_type":        row["formatted_work_type"].strip(),
            })

    print(f"  {len(records)} jobs loaded  ({skipped} skipped — no text)")

    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved → {OUTPUT}  ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
