import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_ALLOWED_CHAR_RE = re.compile(
    r"[A-Za-z0-9\u4e00-\u9fff\u3000-\u303f\uFF00-\uFFEF。，、；：？！「」『』（）()《》“”\"'’‘—–\-…·•\s]"
)


def log_step(message: str) -> None:
    print(f"\n--- {message} ---")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def is_low_info_line(line: str) -> bool:
    if not line:
        return True
    lower = line.lower().strip()
    if lower in {"note", "notes", "page"}:
        return True
    if re.fullmatch(r"\d+", line):
        return True
    if len(line) <= 2:
        return True
    unique_ratio = len(set(line)) / max(1, len(line))
    if unique_ratio < 0.2 and len(line) > 5:
        return True
    alnum_count = len(re.findall(r"[\w]", line, flags=re.UNICODE))
    if alnum_count / max(1, len(line)) < 0.3:
        return True
    if re.search(r"[\-_=~]{4,}", line):
        return True
    if re.search(r"[\*•·]{3,}", line):
        return True
    return False


def allowed_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    allowed = sum(1 for ch in text if _ALLOWED_CHAR_RE.match(ch))
    return allowed / max(1, len(text))


def is_noisy_line(line: str) -> bool:
    if "\ufffd" in line:
        return True
    if re.search(r"[\x00-\x08\x0b-\x1f]", line):
        return True
    if len(line) >= 8 and allowed_char_ratio(line) < 0.6:
        return True
    return False


def estimate_tokens(text: str) -> int:
    tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
    return len(tokens)


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokens = re.findall(r"\w+|[^\s\w]|\s+", text, flags=re.UNICODE)
    count = 0
    kept: List[str] = []
    for token in tokens:
        if token.isspace():
            kept.append(token)
            continue
        count += 1
        if count > max_tokens:
            break
        kept.append(token)
    return "".join(kept).strip()


def text_head(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def file_sha1(path: str) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_lines(text: str) -> List[str]:
    lines = [normalize_line(line) for line in text.splitlines()]
    return [line for line in lines if line]


def filter_lines(lines: Iterable[str]) -> List[str]:
    filtered: List[str] = []
    for line in lines:
        if is_low_info_line(line):
            continue
        if is_noisy_line(line):
            continue
        if re.search(r"[^\w\s\u4e00-\u9fff]", line) and len(line) <= 3:
            continue
        filtered.append(line)
    return filtered


def detect_section_title(line: str) -> Optional[str]:
    if re.match(r"^第\s*\d+\s*[章節篇].*", line):
        return line
    if re.match(r"^[0-9]+\.[0-9]+(\.[0-9]+)?\s+.+", line):
        return line
    if re.match(r"^[•▌■◆◆▍▶►]\s*\S+", line):
        return line
    if line.isupper() and 3 <= len(line) <= 30:
        return line
    return None


def line_hash(line: str) -> str:
    normalized = normalize_line(line)
    normalized = re.sub(r"\d", "", normalized)
    return hashlib.sha1(normalized.lower().encode("utf-8")).hexdigest()


def remove_repeated_lines(pages_lines: List[List[str]], threshold: int) -> List[List[str]]:
    line_counts: Dict[str, int] = {}
    for lines in pages_lines:
        unique = {line_hash(line) for line in lines if line}
        for h in unique:
            line_counts[h] = line_counts.get(h, 0) + 1

    repeated: Set[str] = {h for h, count in line_counts.items() if count >= threshold}
    cleaned_pages: List[List[str]] = []
    for lines in pages_lines:
        cleaned = [line for line in lines if line_hash(line) not in repeated]
        cleaned_pages.append(cleaned)
    return cleaned_pages


def dedupe_lines(lines: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return result


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def parse_page_ranges(value: Optional[str]) -> Optional[Set[int]]:
    if not value:
        return None
    pages: Set[int] = set()
    parts = [v.strip() for v in value.split(",") if v.strip()]
    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            for idx in range(min(start, end), max(start, end) + 1):
                pages.add(idx)
        else:
            try:
                pages.add(int(part))
            except ValueError:
                continue
    return pages if pages else None


def detect_printed_page(line: str) -> Optional[int]:
    line = line.strip()
    match = re.match(r"^第\s*(\d+)\s*頁$", line)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"\d{1,4}", line):
        return int(line)
    return None


def collect_printed_page(lines: Iterable[str]) -> Tuple[Optional[int], List[str]]:
    printed_page = None
    cleaned: List[str] = []
    for line in lines:
        candidate = detect_printed_page(line)
        if candidate is not None and printed_page is None:
            printed_page = candidate
            continue
        cleaned.append(line)
    return printed_page, cleaned
