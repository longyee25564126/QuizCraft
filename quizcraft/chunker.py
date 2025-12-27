import re
from typing import Dict, List, Optional, Tuple

from quizcraft.utils import (
    detect_section_title,
    estimate_tokens,
    log_step,
    normalize_lines,
)


def _tail_by_tokens(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    tokens = re.findall(r"\w+|[^\s\w]|\s+", text, flags=re.UNICODE)
    count = 0
    kept: List[str] = []
    for token in reversed(tokens):
        if token.isspace():
            kept.append(token)
            continue
        count += 1
        if count > overlap_tokens:
            break
        kept.append(token)
    return "".join(reversed(kept)).strip()


def _split_long_line(line: str, target_tokens: int) -> List[str]:
    tokens = re.findall(r"\w+|[^\s\w]|\s+", line, flags=re.UNICODE)
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for token in tokens:
        if token.isspace():
            current.append(token)
            continue
        if count + 1 > target_tokens:
            chunk_text = "".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current = [token]
            count = 1
            continue
        current.append(token)
        count += 1
    final = "".join(current).strip()
    if final:
        chunks.append(final)
    return chunks


def _chunk_lines(lines: List[str], target_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    current_lines: List[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if line_tokens > target_tokens:
            for part in _split_long_line(line, target_tokens):
                if current_lines:
                    chunks.append("\n".join(current_lines).strip())
                    current_lines = []
                    current_tokens = 0
                chunks.append(part.strip())
            continue

        if current_tokens + line_tokens > target_tokens and current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
            overlap_text = _tail_by_tokens(chunk_text, overlap_tokens)
            current_lines = [overlap_text] if overlap_text else []
            current_tokens = estimate_tokens(overlap_text)

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def _split_sections(lines: List[str], current_title: Optional[str]) -> Tuple[List[Tuple[Optional[str], List[str]]], Optional[str]]:
    sections: List[Tuple[Optional[str], List[str]]] = []
    buffer: List[str] = []
    title = current_title
    for line in lines:
        new_title = detect_section_title(line)
        if new_title:
            if buffer:
                sections.append((title, buffer))
                buffer = []
            title = new_title
            continue
        buffer.append(line)
    if buffer:
        sections.append((title, buffer))
    return sections, title


def chunk_pages(
    pages: List[Dict[str, object]],
    chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
) -> List[Dict[str, str]]:
    log_step("Chunk text")
    chunks: List[Dict[str, str]] = []
    section_title: Optional[str] = None

    for page in pages:
        page_no = page["page"]
        lines = page.get("lines")
        if not isinstance(lines, list) or not lines:
            lines = normalize_lines(page.get("text", ""))

        sections, section_title = _split_sections(lines, section_title)
        chunk_index = 1
        for title, section_lines in sections:
            if not section_lines:
                continue
            chunk_texts = _chunk_lines(section_lines, chunk_tokens, overlap_tokens)
            for chunk_text in chunk_texts:
                if title:
                    chunk_text = f"{title}\n{chunk_text}".strip()
                if estimate_tokens(chunk_text) < min_chunk_tokens:
                    continue
                chunks.append(
                    {
                        "page": page_no,
                        "pdf_page_index": page_no,
                        "printed_page": page.get("printed_page"),
                        "chunk_id": f"p{page_no}_c{chunk_index}",
                        "section_title": title or "",
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

    print(f"Created {len(chunks)} chunks.")
    return chunks
