"""
convert_pdfs.py

Extracts plain text from the resume dataset and writes resume_texts.json.

Primary source: Resume.csv (Resume_str column — already clean text).
Fallback:       PDF files under resume-dataset/data/data/<CATEGORY>/<ID>.pdf
                using pdfplumber (preferred) or pypdf.

Run:
    python convert_pdfs.py            # fast path — reads from CSV
    python convert_pdfs.py --from-pdf # slow path — extracts from PDFs directly
"""

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "resume-dataset/Resume/Resume.csv"
PDF_DIR  = PROJECT_ROOT / "resume-dataset/data/data"
OUTPUT   = PROJECT_ROOT / "resume_texts.json"


def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)        # collapse horizontal whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)     # collapse excessive blank lines
    return text.strip()


# ── CSV path (fast) ──────────────────────────────────────────────────────────

def load_from_csv() -> list[dict]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    records = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            text = clean_text(row["Resume_str"])
            if text:
                records.append({
                    "id":       row["ID"],
                    "category": row["Category"],
                    "text":     text,
                    "source":   "csv",
                })
    return records


# ── PDF path (direct extraction) ─────────────────────────────────────────────

def _extract_with_pdfplumber(path: Path) -> str:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _extract_with_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pdf_text(path: Path) -> str:
    for extractor in (_extract_with_pdfplumber, _extract_with_pypdf):
        try:
            return extractor(path)
        except ImportError:
            continue
    raise RuntimeError(
        "No PDF library found. Install one:\n"
        "  pip install pdfplumber\n"
        "  pip install pypdf"
    )


def load_from_pdfs() -> list[dict]:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")
    records = []
    pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
    total = len(pdf_paths)
    for i, pdf_path in enumerate(pdf_paths, 1):
        category = pdf_path.parent.name
        text = clean_text(extract_pdf_text(pdf_path))
        if text:
            records.append({
                "id":       pdf_path.stem,
                "category": category,
                "text":     text,
                "source":   "pdf",
            })
        print(f"  [{i}/{total}] {pdf_path.name} ({category})", end="\r", flush=True)
    print()
    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert resume data to plain text JSON.")
    parser.add_argument("--from-pdf", action="store_true",
                        help="Extract text directly from PDFs instead of the CSV.")
    args = parser.parse_args()

    if args.from_pdf:
        print(f"Extracting text from PDFs in {PDF_DIR} ...")
        records = load_from_pdfs()
    else:
        print(f"Loading resume text from {CSV_PATH} ...")
        records = load_from_csv()

    print(f"  {len(records)} resumes loaded")

    categories = {}
    for r in records:
        categories[r["category"]] = categories.get(r["category"], 0) + 1
    print(f"  {len(categories)} categories: {', '.join(sorted(categories))}")

    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved → {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
