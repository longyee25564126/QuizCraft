from typing import Any, Dict, List

from pypdf import PdfReader

from quizcraft.config import Settings
from quizcraft.utils import (
    clean_text,
    collect_printed_page,
    dedupe_lines,
    filter_lines,
    log_step,
    normalize_lines,
    remove_repeated_lines,
)


def _should_ocr(lines: List[str]) -> bool:
    if not lines:
        return True
    text = " ".join(lines)
    return len(text.strip()) < 30


def _run_ocr(path: str, page_number: int, settings: Settings) -> str:
    if not settings.enable_ocr:
        return ""
    if settings.ocr_engine == "easyocr":
        try:
            import easyocr  # type: ignore
            from pdf2image import convert_from_path  # type: ignore
        except Exception as exc:
            print(f"OCR skipped (easyocr unavailable): {exc}")
            return ""
        images = convert_from_path(path, first_page=page_number, last_page=page_number)
        if not images:
            return ""
        reader = easyocr.Reader([settings.ocr_lang], gpu=False)
        results = reader.readtext(images[0], detail=0)
        return "\n".join(results)

    try:
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
    except Exception as exc:
        print(f"OCR skipped (tesseract unavailable): {exc}")
        return ""

    images = convert_from_path(path, first_page=page_number, last_page=page_number)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=settings.ocr_lang)


def ingest_pdf(path: str, settings: Settings) -> List[Dict[str, Any]]:
    log_step(f"Ingest PDF: {path}")
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        lines = filter_lines(normalize_lines(text))
        printed_page, lines = collect_printed_page(lines)
        lines = dedupe_lines(lines)
        return [
            {
                "page": 1,
                "text": "\n".join(lines),
                "printed_page": printed_page,
                "lines": lines,
            }
        ]

    reader = PdfReader(path)
    pages_raw: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = filter_lines(normalize_lines(text))
        printed_page, lines = collect_printed_page(lines)
        lines = dedupe_lines(lines)
        pages_raw.append(
            {
                "page": idx,
                "printed_page": printed_page,
                "lines": lines,
                "text": "\n".join(lines),
            }
        )

    threshold = settings.header_footer_threshold
    if threshold <= 0:
        threshold = max(3, int(len(pages_raw) * 0.4))
    cleaned = remove_repeated_lines([page["lines"] for page in pages_raw], threshold)

    pages: List[Dict[str, Any]] = []
    for raw, lines in zip(pages_raw, cleaned):
        text = "\n".join(lines).strip()
        if _should_ocr(lines):
            ocr_text = _run_ocr(path, raw["page"], settings)
            if ocr_text:
                ocr_lines = filter_lines(normalize_lines(ocr_text))
                _, ocr_lines = collect_printed_page(ocr_lines)
                ocr_lines = dedupe_lines(ocr_lines)
                lines = ocr_lines
                text = "\n".join(lines).strip()
        pages.append(
            {
                "page": raw["page"],
                "text": clean_text(text),
                "printed_page": raw["printed_page"],
                "lines": lines,
            }
        )

    print(f"Loaded {len(pages)} pages.")
    return pages
