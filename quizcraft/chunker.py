from typing import Dict, List

from quizcraft.utils import clean_text, log_step


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            cut = cleaned.rfind(" ", start + int(chunk_size * 0.5), end)
            if cut == -1:
                cut = end
            end = cut
        if end <= start:
            break
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pages(pages: List[Dict[str, str]], chunk_size: int, overlap: int) -> List[Dict[str, str]]:
    log_step("Chunk text")
    chunks: List[Dict[str, str]] = []
    for page in pages:
        page_no = page["page"]
        page_chunks = chunk_text(page["text"], chunk_size, overlap)
        for idx, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "page": page_no,
                "chunk_id": f"p{page_no}_c{idx}",
                "text": chunk,
            })
    print(f"Created {len(chunks)} chunks.")
    return chunks
