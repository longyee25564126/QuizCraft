from typing import Dict, List

from pypdf import PdfReader

from quizcraft.utils import clean_text, log_step


def ingest_pdf(path: str) -> List[Dict[str, str]]:
    log_step(f"Ingest PDF: {path}")
    reader = PdfReader(path)
    pages: List[Dict[str, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": idx, "text": clean_text(text)})
    print(f"Loaded {len(pages)} pages.")
    return pages
