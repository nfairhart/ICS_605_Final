"""
pdf_parser.py

Extract plain text from an uploaded PDF file object (BytesIO or file-like).
"""

import re
import io


def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes. Tries pdfplumber first, falls back to pypdf."""
    buf = io.BytesIO(file_bytes)

    try:
        import pdfplumber
        buf.seek(0)
        with pdfplumber.open(buf) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return clean_text(text)
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        buf.seek(0)
        reader = PdfReader(buf)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return clean_text(text)
    except ImportError:
        pass

    raise RuntimeError("No PDF library available. Install pdfplumber: pip install pdfplumber")
