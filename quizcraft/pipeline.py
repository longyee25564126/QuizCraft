import hashlib
import json
import os
import random
import re
from typing import Dict, List, Tuple

from quizcraft.chunker import chunk_pages
from quizcraft.config import Settings
from quizcraft.ingest import ingest_pdf
from quizcraft.ollama_client import OllamaClient
from quizcraft.prompts import (
    CONCEPT_PROMPT,
    MAP_SUMMARY_PROMPT,
    QUESTION_PROMPT,
    REDUCE_SUMMARY_PROMPT,
    VERIFY_PROMPT,
)
from quizcraft.retrieval import build_index, search_index, select_chunks
from quizcraft.schemas import (
    Concept,
    EvidenceQuote,
    MiniSummary,
    Question,
    QuizOutput,
    SummaryBlock,
    SummarySection,
    normalize_concepts,
    normalize_mini_summary,
    normalize_question,
)
from quizcraft.utils import (
    allowed_char_ratio,
    detect_section_title,
    estimate_tokens,
    file_sha1,
    is_low_info_line,
    is_noisy_line,
    log_step,
    normalize_line,
    text_head,
    trim_to_tokens,
)

_CITATION_RE = re.compile(r"p\d+_c\d+")
_META_QUESTION_RE = re.compile(
    r"(哪一頁|哪一段|頁碼|頁號|頁面|page\b|chunk(?:_id)?|出處|來源|段落|p\d+_c\d+)",
    re.IGNORECASE,
)

_BANNED_MCQ_RE = re.compile(r"(all of the above|以上皆是|以上皆對|以上皆為|以上皆正確)", re.IGNORECASE)
_QUESTION_WORD_RE = re.compile(r"(什麼|為何|如何|哪|幾|多少|是否|能否|可否|嗎|呢)")

_INCOMPLETE_SUFFIXES = (
    "並",
    "以及",
    "而且",
    "且",
    "並且",
    "引進",
    "推行",
    "包含",
    "包括",
    "例如",
    "如",
    "等",
    "等等",
    "並引進",
    "並將",
)


def _format_evidence(chunks: List[Dict[str, str]], max_tokens: int, max_chars: int) -> str:
    entries = []
    total_tokens = 0
    total_chars = 0
    for chunk in chunks:
        remaining_tokens = max_tokens - total_tokens
        if remaining_tokens <= 0:
            break
        text = trim_to_tokens(chunk["text"], remaining_tokens)
        entry = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "printed_page": chunk.get("printed_page"),
            "section_title": chunk.get("section_title", ""),
            "text": text,
        }
        serialized = json.dumps(entry, ensure_ascii=False)
        if max_chars > 0 and total_chars + len(serialized) > max_chars and entries:
            break
        entries.append(entry)
        total_tokens += estimate_tokens(text)
        total_chars += len(serialized)
        if max_chars > 0 and total_chars >= max_chars:
            break
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _format_mini_summaries(mini_summaries: List[MiniSummary], max_tokens: int, max_chars: int) -> str:
    entries = []
    total_tokens = 0
    total_chars = 0
    for summary in mini_summaries:
        entry = {
            "mini_summary": summary["mini_summary"],
            "keywords": summary["keywords"],
            "citations": summary["citations"],
        }
        serialized = json.dumps(entry, ensure_ascii=False)
        tokens = estimate_tokens(serialized)
        if max_tokens > 0 and total_tokens + tokens > max_tokens and entries:
            break
        if max_chars > 0 and total_chars + len(serialized) > max_chars and entries:
            break
        entries.append(entry)
        total_tokens += tokens
        total_chars += len(serialized)
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text))

def _is_meta_question(text: str) -> bool:
    if not text:
        return False
    return bool(_META_QUESTION_RE.search(text))

def _is_question_sentence(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if stripped.endswith(("?", "？")):
        return True
    if re.search(r"(嗎|呢)$", stripped):
        return True
    if _QUESTION_WORD_RE.search(stripped):
        return True
    return False


def _fix_tf_question(text: str) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = cleaned.rstrip("？?")
    cleaned = re.sub(r"^(請問|請回答)", "", cleaned).strip()
    cleaned = re.sub(r"(是否|是不是|能否|可否)", "", cleaned)
    cleaned = re.sub(r"(嗎|呢)$", "", cleaned).strip()
    if not cleaned:
        return None
    if _is_question_sentence(cleaned):
        return None
    return cleaned


def _has_banned_mcq_choice(choices: List[str]) -> bool:
    return any(_BANNED_MCQ_RE.search(choice or "") for choice in choices)


def _chunk_body_lines(chunk: Dict[str, str]) -> List[str]:
    lines = [normalize_line(line) for line in str(chunk.get("text", "")).splitlines()]
    lines = [line for line in lines if line]
    if lines:
        section_title = chunk.get("section_title") or ""
        if detect_section_title(lines[0]) or (section_title and lines[0] == section_title):
            lines = lines[1:]
    cleaned: List[str] = []
    for line in lines:
        if is_low_info_line(line) or is_noisy_line(line):
            continue
        if detect_section_title(line):
            continue
        cleaned.append(line)
    return cleaned


def _is_low_info_chunk(chunk: Dict[str, str]) -> bool:
    lines = _chunk_body_lines(chunk)
    if not lines:
        return True
    body = " ".join(lines)
    if len(body) < 40:
        return True
    if allowed_char_ratio(body) < 0.6:
        return True
    return False


def _extract_quote_from_chunk(chunk: Dict[str, str], min_len: int = 40, max_len: int = 80) -> str:
    lines = _chunk_body_lines(chunk)
    for line in lines:
        if len(line) < min_len:
            continue
        if allowed_char_ratio(line) < 0.7:
            continue
        return line[:max_len]
    for line in lines:
        if len(line) < 20:
            continue
        if allowed_char_ratio(line) < 0.7:
            continue
        return line[:max_len]
    body = " ".join(lines).strip()
    if len(body) >= min_len and allowed_char_ratio(body) >= 0.7:
        return body[:max_len]
    if len(body) >= 20 and allowed_char_ratio(body) >= 0.7:
        return body[:max_len]
    return ""


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"[;；]+", "。", text)
    text = text.replace("\n", "。")
    parts = re.split(r"[。！？]+", text)
    return [part.strip() for part in parts if part.strip()]


def _is_incomplete_sentence(sentence: str) -> bool:
    sentence = sentence.strip()
    if not sentence:
        return True
    if sentence.endswith(_INCOMPLETE_SUFFIXES):
        return True
    if sentence.endswith(("(", "（", "：", "，", "、", "；", "-", "—", "…")):
        return True
    if sentence.count("（") > sentence.count("）"):
        return True
    if sentence.count("(") > sentence.count(")"):
        return True
    return False


def _normalize_paragraph(text: str, min_sentences: int, max_sentences: int, fallback: List[str]) -> str:
    sentences = _split_sentences(text)
    sentences = [s for s in sentences if not _is_incomplete_sentence(s)]
    if len(sentences) < min_sentences:
        for extra in fallback:
            extra = extra.strip()
            if not extra or _is_incomplete_sentence(extra):
                continue
            if extra in sentences:
                continue
            sentences.append(extra)
            if len(sentences) >= min_sentences:
                break
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    if not sentences:
        return ""
    return "。".join(sentences) + "。"


def _sentences_from_mini_summaries(mini_summaries: List[MiniSummary]) -> List[str]:
    sentences: List[str] = []
    for ms in mini_summaries:
        for sentence in _split_sentences(ms.get("mini_summary", "")):
            if not sentence or _is_incomplete_sentence(sentence):
                continue
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences


def _normalize_keypoints(raw_keypoints: List[str], fallback_sentences: List[str]) -> List[str]:
    cleaned: List[str] = []
    for kp in raw_keypoints:
        kp = re.sub(r"p\d+_c\d+", "", kp or "").strip()
        kp = re.sub(r"\(\s*\)", "", kp).strip()
        kp = kp.rstrip("。；; ")
        if not kp or _is_incomplete_sentence(kp):
            continue
        if kp not in cleaned:
            cleaned.append(kp)
    if len(cleaned) < 5:
        for sentence in fallback_sentences:
            if len(cleaned) >= 5:
                break
            candidate = sentence.strip().rstrip("。")
            if not candidate or _is_incomplete_sentence(candidate):
                continue
            if candidate not in cleaned:
                cleaned.append(candidate)
    return cleaned[:8]


def _citations_for_text(
    query: str,
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
    min_count: int = 2,
    max_count: int = 4,
) -> List[Dict[str, str | int]]:
    if not query:
        return []
    matches = search_index(
        query,
        chunks,
        embeddings,
        lambda text: client.embed(settings.embed_model, text),
        top_k=max(8, max_count * 2),
    )
    matches = _filter_low_info_chunks(matches)
    citations: List[Dict[str, str | int]] = []
    seen_pages = set()
    for chunk in matches:
        if chunk["page"] in seen_pages:
            continue
        citations.append({"page": chunk["page"], "chunk_id": chunk["chunk_id"]})
        seen_pages.add(chunk["page"])
        if len(citations) >= max_count:
            break
    if len(citations) < min_count:
        for chunk in matches:
            if len(citations) >= min_count:
                break
            citation = {"page": chunk["page"], "chunk_id": chunk["chunk_id"]}
            if citation in citations:
                continue
            citations.append(citation)
    return citations[:max_count]


def _normalize_sections(
    raw_sections: List[Dict[str, object]],
    selected_chunks: List[Dict[str, str]],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
    mini_by_chunk_id: Dict[str, MiniSummary],
    target_sections: int,
) -> List[SummarySection]:
    sections: List[SummarySection] = []
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    for raw in raw_sections:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        summary = str(raw.get("summary", "")).strip()
        query = f"{title} {summary}".strip()
        matches = search_index(
            query or title,
            chunks,
            embeddings,
            lambda text: client.embed(settings.embed_model, text),
            top_k=6,
        )
        matches = _filter_low_info_chunks(matches)
        fallback_sentences: List[str] = []
        for chunk in matches:
            ms = mini_by_chunk_id.get(chunk["chunk_id"])
            if ms:
                fallback_sentences.extend(_split_sentences(ms.get("mini_summary", "")))
        summary = _normalize_paragraph(summary, 2, 4, fallback_sentences)
        citations = _normalize_citations(raw.get("citations"))
        citations = [c for c in citations if c.get("chunk_id") in chunk_lookup]
        if len(citations) < 2:
            citations = _citations_for_text(query or summary, chunks, embeddings, client, settings)
        deduped: List[Dict[str, str | int]] = []
        seen_pages = set()
        for citation in citations:
            if citation["page"] in seen_pages:
                continue
            deduped.append(citation)
            seen_pages.add(citation["page"])
            if len(deduped) >= 4:
                break
        citations = deduped
        if not title:
            if matches:
                title = matches[0].get("section_title", "") or f"第{matches[0]['page']}頁主題"
            else:
                title = "章節重點"
        if not summary:
            summary = _normalize_paragraph("", 2, 4, fallback_sentences)
        if not summary:
            continue
        sections.append({"title": title, "summary": summary, "citations": citations})
        if len(sections) >= target_sections:
            break
    return sections


def _build_section_groups(selected_chunks: List[Dict[str, str]], target_sections: int) -> List[Tuple[str, List[Dict[str, str]]]]:
    pages = sorted({chunk["page"] for chunk in selected_chunks})
    if not pages:
        return []
    bucket_size = max(1, (len(pages) + target_sections - 1) // target_sections)
    groups: List[Tuple[str, List[Dict[str, str]]]] = []
    for idx in range(0, len(pages), bucket_size):
        bucket_pages = pages[idx : idx + bucket_size]
        bucket_chunks = [
            chunk for chunk in selected_chunks if chunk["page"] in bucket_pages and not _is_low_info_chunk(chunk)
        ]
        if not bucket_chunks:
            continue
        title = bucket_chunks[0].get("section_title") or f"第{bucket_pages[0]}-{bucket_pages[-1]}頁重點"
        groups.append((title, bucket_chunks))
    return groups


def _ensure_section_coverage(
    sections: List[SummarySection],
    selected_chunks: List[Dict[str, str]],
) -> List[SummarySection]:
    if not sections:
        return sections
    pages_all = sorted({chunk["page"] for chunk in selected_chunks})
    if not pages_all:
        return sections
    required = max(1, int(len(pages_all) * 0.6))
    covered = {citation["page"] for section in sections for citation in section.get("citations", [])}
    if len(covered) >= required:
        return sections
    missing_pages = [page for page in pages_all if page not in covered]
    page_chunks = {}
    for page in missing_pages:
        for chunk in selected_chunks:
            if chunk["page"] == page and not _is_low_info_chunk(chunk):
                page_chunks[page] = chunk
                break
    for idx, page in enumerate(missing_pages):
        chunk = page_chunks.get(page)
        if not chunk:
            continue
        section = sections[idx % len(sections)]
        existing_pages = {c["page"] for c in section.get("citations", [])}
        if page in existing_pages:
            continue
        if len(section.get("citations", [])) >= 4:
            continue
        section["citations"].append({"page": chunk["page"], "chunk_id": chunk["chunk_id"]})
    return sections


def _validate_summary_block(summary: SummaryBlock, min_unique_pages: int, min_citations: int) -> bool:
    overview = summary.get("overview", "")
    if len(_split_sentences(overview)) < 2:
        return False
    sections = summary.get("sections", [])
    if len(sections) < 3:
        return False
    keypoints = summary.get("keypoints", [])
    if len(keypoints) < 5:
        return False
    for section in sections:
        if len(_split_sentences(section.get("summary", ""))) < 2:
            return False
        citations = section.get("citations", [])
        if len(citations) < min_citations:
            return False
        pages = [c["page"] for c in citations if isinstance(c, dict) and c.get("page") is not None]
        if len(set(pages)) < min_unique_pages:
            return False
    return True


def _build_summary_block_from_mini(
    mini_summaries: List[MiniSummary],
    selected_chunks: List[Dict[str, str]],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
) -> SummaryBlock:
    fallback_sentences = _sentences_from_mini_summaries(mini_summaries)
    overview = _normalize_paragraph("", 2, 3, fallback_sentences)
    target_sections = min(6, max(3, (len({c['page'] for c in selected_chunks}) + 4) // 5))
    total_pages = len({c['page'] for c in selected_chunks})
    min_unique_pages = 2 if total_pages >= 2 else 1
    min_citations = 2 if total_pages >= 2 else 1
    groups = _build_section_groups(selected_chunks, target_sections)
    sections: List[SummarySection] = []
    mini_by_chunk_id = {ms["chunk_id"]: ms for ms in mini_summaries}
    for title, group in groups:
        group_sentences: List[str] = []
        for chunk in group:
            ms = mini_by_chunk_id.get(chunk["chunk_id"])
            if ms:
                group_sentences.extend(_split_sentences(ms.get("mini_summary", "")))
        summary = _normalize_paragraph("", 2, 4, group_sentences)
        citations = _citations_for_text(title + " " + summary, chunks, embeddings, client, settings)
        sections.append({"title": title, "summary": summary, "citations": citations})
        if len(sections) >= target_sections:
            break
    sections = _ensure_section_coverage(sections, selected_chunks)
    keypoints = _normalize_keypoints([], fallback_sentences)
    return {"overview": overview, "sections": sections, "keypoints": keypoints}

def _citation_tag(citation: Dict[str, str | int]) -> str:
    return f"p{citation['page']}:{citation['chunk_id']}"

def _normalize_citations(raw) -> List[Dict[str, str | int]]:
    citations: List[Dict[str, str | int]] = []
    if not isinstance(raw, list):
        return citations
    for item in raw:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        chunk_id = item.get("chunk_id") or item.get("chunkId")
        if page is None or chunk_id is None:
            continue
        try:
            page_num = int(page)
        except (ValueError, TypeError):
            continue
        citations.append({"page": page_num, "chunk_id": str(chunk_id)})
    return citations


def _attach_citations_to_summary(summary: str, citations: List[Dict[str, str | int]], settings: Settings) -> str:
    if _has_citation(summary) or not citations:
        return summary
    tags = [_citation_tag(c) for c in citations][:3]
    citation_text = f"（參考 {', '.join(tags)}）"
    if len(summary) + len(citation_text) > settings.summary_max_chars:
        summary = summary[: max(0, settings.summary_max_chars - len(citation_text))]
    return f"{summary}{citation_text}"


def _keypoint_best_citation(keypoint: str, mini_summaries: List[MiniSummary]) -> Dict[str, str | int] | None:
    keypoint = keypoint.strip()
    best_score = -1
    best_citation = None
    for ms in mini_summaries:
        score = 0
        for kw in ms.get("keywords", []):
            if kw and kw in keypoint:
                score += 2
        if ms.get("mini_summary") and keypoint and keypoint in ms["mini_summary"]:
            score += 1
        if score > best_score and ms.get("citations"):
            best_score = score
            best_citation = ms["citations"][0]
    return best_citation


def _attach_citations_to_keypoints(keypoints: List[str], mini_summaries: List[MiniSummary]) -> List[str]:
    updated: List[str] = []
    for kp in keypoints:
        if _has_citation(kp):
            updated.append(kp)
            continue
        citation = _keypoint_best_citation(kp, mini_summaries)
        if citation:
            updated.append(f"{kp} ({_citation_tag(citation)})")
        else:
            updated.append(kp)
    return updated


def _validate_summary(summary: str, keypoints: List[str], settings: Settings) -> bool:
    if not summary:
        return False
    length = len(summary)
    if length < settings.summary_min_chars or length > settings.summary_max_chars:
        return False
    if len(keypoints) < 3 or len(keypoints) > 5:
        return False
    return True


def _fallback_summary_from_mini(
    mini_summaries: List[MiniSummary],
    settings: Settings,
) -> Tuple[str, List[str], List[Dict[str, str | int]]]:
    summary_parts: List[str] = []
    citations: List[Dict[str, str | int]] = []

    for ms in mini_summaries:
        text = ms.get("mini_summary", "").strip()
        if text:
            summary_parts.append(text)
        citations.extend(ms.get("citations", []))
        current = "；".join(summary_parts)
        if len(current) >= settings.summary_min_chars:
            break

    summary = "；".join(summary_parts).strip("；")
    if len(summary) > settings.summary_max_chars:
        summary = summary[: settings.summary_max_chars]

    keypoints: List[str] = []
    seen = set()
    for ms in mini_summaries:
        for kw in ms.get("keywords", []):
            kw = kw.strip()
            if kw and kw not in seen:
                keypoints.append(kw)
                seen.add(kw)
            if len(keypoints) >= 5:
                break
        if len(keypoints) >= 5:
            break

    if len(keypoints) < 3:
        for ms in mini_summaries:
            snippet = ms.get("mini_summary", "").strip()
            if not snippet:
                continue
            if len(snippet) > 30:
                snippet = snippet[:30].rstrip("，。；") + "..."
            if snippet not in seen:
                keypoints.append(snippet)
                seen.add(snippet)
            if len(keypoints) >= 3:
                break

    keypoints = keypoints[:5]

    deduped: List[Dict[str, str | int]] = []
    seen_ids = set()
    for citation in citations:
        key = f"{citation.get('page')}:{citation.get('chunk_id')}"
        if key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(citation)

    if not deduped and mini_summaries:
        deduped = mini_summaries[0].get("citations", [])

    return summary, keypoints, deduped


def _validate_question(question: Question, settings: Settings) -> bool:
    q_type = question.get("type", "").lower()
    if q_type not in {t.lower() for t in settings.question_types}:
        return False
    if not question.get("question") or not question.get("answer") or not question.get("rationale"):
        return False
    if _is_meta_question(question.get("question", "")):
        return False
    if not question.get("citations"):
        return False
    quotes = question.get("evidence_quotes")
    if not quotes:
        return False
    for quote in quotes:
        text = str(quote.get("quote", "")).strip()
        if len(text) < 20 or len(text) > 80:
            return False
        if allowed_char_ratio(text) < 0.7:
            return False
        if is_noisy_line(text):
            return False
    if q_type == "mcq":
        choices = question.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 4:
            return False
        if _has_banned_mcq_choice(choices):
            return False
        for idx, choice in enumerate(choices):
            prefix = f"{chr(ord('A') + idx)} "
            if not str(choice).startswith(prefix):
                return False
        correct = (question.get("correct_option") or question.get("answer", "")).upper()
        if correct not in {"A", "B", "C", "D"}:
            return False
        if question.get("answer", "").upper() != correct:
            return False
    if q_type == "tf":
        if question.get("answer") not in {"true", "false"}:
            return False
        if _is_question_sentence(question.get("question", "")):
            return False
    if q_type == "short":
        if question.get("answer", "").lower() in {"true", "false"}:
            return False
    if q_type == "calc":
        steps = question.get("step_by_step", [])
        if not isinstance(steps, list) or not steps:
            return False
        if not question.get("final_answer"):
            return False
    return True


def _select_chunk_set(
    pages: List[Dict[str, str]],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
) -> List[Dict[str, str]]:
    page_count = len(pages)
    use_selector = (
        page_count >= settings.long_doc_threshold_pages
        or len(chunks) >= settings.selector_chunk_threshold
        or len(chunks) > settings.max_chunks
    )

    if not use_selector:
        return chunks

    top_k = min(settings.top_k_chunks, settings.max_chunks, len(chunks))
    return select_chunks(
        chunks,
        embeddings,
        lambda text: client.embed(settings.embed_model, text),
        top_k=top_k,
        seed=settings.seed,
    )


def _ensure_evidence_quotes(
    question: Question,
    evidence_chunks: List[Dict[str, str]],
) -> None:
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in evidence_chunks}
    quotes: List[EvidenceQuote] = []

    citations = question.get("citations", [])
    for citation in citations:
        chunk = chunk_lookup.get(citation.get("chunk_id"))
        if not chunk:
            continue
        quote = _extract_quote_from_chunk(chunk)
        if quote:
            quotes.append({"page": chunk["page"], "chunk_id": chunk["chunk_id"], "quote": quote})
            break

    if not quotes:
        for chunk in evidence_chunks:
            quote = _extract_quote_from_chunk(chunk)
            if quote:
                quotes.append({"page": chunk["page"], "chunk_id": chunk["chunk_id"], "quote": quote})
                break

    question["evidence_quotes"] = quotes


def _ensure_citations(question: Question, evidence_chunks: List[Dict[str, str]]) -> None:
    chunk_ids = {chunk["chunk_id"] for chunk in evidence_chunks}
    citations = [c for c in question.get("citations", []) if c.get("chunk_id") in chunk_ids]
    if not citations and evidence_chunks:
        citations = [{"page": evidence_chunks[0]["page"], "chunk_id": evidence_chunks[0]["chunk_id"]}]
    question["citations"] = citations


def _ensure_rationale_quotes(question: Question) -> None:
    rationale = question.get("rationale", "")
    for quote in question.get("evidence_quotes", []):
        if quote.get("quote") and quote["quote"] in rationale:
            return
    if question.get("evidence_quotes"):
        snippet = question["evidence_quotes"][0]["quote"]
        question["rationale"] = f"{rationale}（原文：{snippet}）".strip()


def _contains_external_reference(text: str, evidence_text: str) -> bool:
    titles = re.findall(r"《[^》]{2,20}》", text)
    for title in titles:
        if title not in evidence_text:
            return True
    return False


def _filter_pages(pages: List[Dict[str, str]], settings: Settings) -> List[Dict[str, str]]:
    filtered = pages
    if settings.pages_filter:
        filtered = [page for page in filtered if page["page"] in settings.pages_filter]
        if not filtered:
            print("Page filter matched no pages; fallback to full document.")
            filtered = pages
    if settings.chapter_filter:
        matches = [
            page
            for page in filtered
            if settings.chapter_filter in page.get("text", "")
            or any(settings.chapter_filter in line for line in page.get("lines", []))
        ]
        if matches:
            filtered = matches
        else:
            print("Chapter filter matched no pages; keeping previous selection.")
    if settings.max_pages is not None:
        filtered = filtered[: settings.max_pages]
    return filtered


def _embedding_cache_path(pdf_path: str, chunks: List[Dict[str, str]], settings: Settings) -> str:
    meta = "|".join(
        [
            file_sha1(pdf_path),
            settings.embed_model,
            str(settings.chunk_tokens),
            str(settings.overlap_tokens),
            str(settings.min_chunk_tokens),
            str(sorted(settings.pages_filter)) if settings.pages_filter else "all",
            str(settings.max_pages or ""),
            settings.chapter_filter or "",
            str(len(chunks)),
        ]
    )
    key = hashlib.sha1(meta.encode("utf-8")).hexdigest()
    return os.path.join(settings.embed_cache_dir, f"embeddings_{key}.json")


def _load_embeddings(cache_path: str, chunks: List[Dict[str, str]]) -> List[List[float]] | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if data.get("chunk_ids") != [chunk["chunk_id"] for chunk in chunks]:
        return None
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(chunks):
        return None
    return embeddings


def _save_embeddings(cache_path: str, chunks: List[Dict[str, str]], embeddings: List[List[float]]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "embeddings": embeddings,
    }
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _build_or_load_embeddings(
    pdf_path: str,
    chunks: List[Dict[str, str]],
    client: OllamaClient,
    settings: Settings,
) -> List[List[float]]:
    if settings.embed_cache_enabled:
        cache_path = _embedding_cache_path(pdf_path, chunks, settings)
        cached = _load_embeddings(cache_path, chunks)
        if cached:
            print("Loaded embeddings from cache.")
            return cached
        embeddings = build_index(chunks, lambda text: client.embed(settings.embed_model, text))
        _save_embeddings(cache_path, chunks, embeddings)
        return embeddings
    return build_index(chunks, lambda text: client.embed(settings.embed_model, text))


def map_summarize(
    client: OllamaClient,
    settings: Settings,
    chunks: List[Dict[str, str]],
) -> List[MiniSummary]:
    log_step("Map summarize")
    summaries: List[MiniSummary] = []
    for chunk in chunks:
        chunk_text = trim_to_tokens(chunk["text"], settings.chunk_tokens)
        prompt = MAP_SUMMARY_PROMPT.format(
            page=chunk["page"],
            chunk_id=chunk["chunk_id"],
            chunk_text=chunk_text,
        )
        try:
            data = client.chat_json(
                settings.chat_model,
                [{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                timeout=settings.chat_timeout,
            )
        except Exception as exc:
            print(f"Map summarize error on {chunk['chunk_id']}: {exc}")
            data = {}

        mini = normalize_mini_summary(data, chunk["page"], chunk["chunk_id"])
        if not mini["mini_summary"]:
            mini["mini_summary"] = text_head(chunk_text, 120)
        summaries.append(mini)
    print(f"Mini summaries: {len(summaries)}")
    return summaries


def reduce_summarize(
    client: OllamaClient,
    settings: Settings,
    mini_summaries: List[MiniSummary],
    selected_chunks: List[Dict[str, str]],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
) -> SummaryBlock:
    log_step("Reduce summarize")
    context = _format_mini_summaries(
        mini_summaries,
        settings.summary_budget_tokens,
        settings.max_input_chars,
    )
    prompt = REDUCE_SUMMARY_PROMPT.format(
        summary_min=settings.summary_min_chars,
        summary_max=settings.summary_max_chars,
        mini_summaries=context,
    )
    attempts = 0
    summary_block: SummaryBlock = {"overview": "", "sections": [], "keypoints": []}
    mini_by_chunk_id = {ms["chunk_id"]: ms for ms in mini_summaries}
    fallback_sentences = _sentences_from_mini_summaries(mini_summaries)
    target_sections = min(6, max(3, (len({c['page'] for c in selected_chunks}) + 4) // 5))
    total_pages = len({c['page'] for c in selected_chunks})
    min_unique_pages = 2 if total_pages >= 2 else 1
    min_citations = 2 if total_pages >= 2 else 1

    while attempts <= settings.summary_retries:
        try:
            data = client.chat_json(
                settings.chat_model,
                [{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                timeout=settings.reduce_timeout,
            )
        except Exception as exc:
            print(f"Reduce summarize error: {exc}")
            data = {}

        overview = str(data.get("overview", "")).strip()
        raw_sections = data.get("sections") if isinstance(data.get("sections"), list) else []
        raw_keypoints = data.get("keypoints") if isinstance(data.get("keypoints"), list) else []
        raw_keypoints = [str(k).strip() for k in raw_keypoints if str(k).strip()]

        overview = _normalize_paragraph(overview, 2, 3, fallback_sentences)
        sections = _normalize_sections(
            raw_sections,
            selected_chunks,
            chunks,
            embeddings,
            client,
            settings,
            mini_by_chunk_id,
            target_sections,
        )
        sections = _ensure_section_coverage(sections, selected_chunks)
        keypoints = _normalize_keypoints(raw_keypoints, fallback_sentences)

        summary_block = {"overview": overview, "sections": sections, "keypoints": keypoints}
        if _validate_summary_block(summary_block, min_unique_pages, min_citations):
            return summary_block
        attempts += 1

    print("Reduce summary fallback triggered.")
    return _build_summary_block_from_mini(
        mini_summaries,
        selected_chunks,
        chunks,
        embeddings,
        client,
        settings,
    )

def _fallback_concepts_from_mini(
    keypoints: List[str],
    mini_summaries: List[MiniSummary],
    settings: Settings,
) -> List[Concept]:
    concepts: List[Concept] = []
    used = set()

    def add_concept(name: str, description: str, citations: List[Dict[str, str | int]]) -> None:
        if not name or name in used:
            return
        used.add(name)
        concepts.append(
            {
                "name": name,
                "description": description,
                "citations": citations,
                "difficulty": "medium",
            }
        )

    first_citation = None
    for ms in mini_summaries:
        if ms.get("citations"):
            first_citation = ms["citations"][0]
            break

    for kp in keypoints:
        add_concept(kp, "", [first_citation] if first_citation else [])
        if len(concepts) >= settings.question_count:
            return concepts

    for ms in mini_summaries:
        citations = ms.get("citations", [])
        for kw in ms.get("keywords", []):
            add_concept(kw, ms.get("mini_summary", "")[:50], citations[:1])
            if len(concepts) >= settings.question_count:
                return concepts
        snippet = ms.get("mini_summary", "").strip()
        if snippet:
            name = snippet[:20].rstrip("，。；") + "..."
            add_concept(name, snippet, citations[:1])
            if len(concepts) >= settings.question_count:
                return concepts

    return concepts


def extract_concepts(
    client: OllamaClient,
    settings: Settings,
    keypoints: List[str],
    mini_summaries: List[MiniSummary],
) -> List[Concept]:
    log_step("Concept extract")
    context = _format_mini_summaries(mini_summaries, settings.summary_budget_tokens, settings.max_input_chars)
    prompt = CONCEPT_PROMPT.format(
        max_concepts=settings.question_count,
        keypoints="\n".join(f"- {kp}" for kp in keypoints),
        mini_summaries=context,
    )
    try:
        data = client.chat_json(
            settings.chat_model,
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
            timeout=settings.chat_timeout,
        )
    except Exception as exc:
        print(f"Concept extract error: {exc}")
        data = {}

    concepts = normalize_concepts(data.get("concepts"))
    concepts = concepts[: settings.question_count]

    if len(concepts) < settings.question_count:
        fallback = _fallback_concepts_from_mini(keypoints, mini_summaries, settings)
        used = {concept["name"] for concept in concepts}
        for concept in fallback:
            if concept["name"] in used:
                continue
            concepts.append(concept)
            used.add(concept["name"])
            if len(concepts) >= settings.question_count:
                break

    print(f"Concepts: {len(concepts)}")
    return concepts


def _collect_evidence_by_citations(
    citations: List[Dict[str, str | int]],
    chunk_lookup: Dict[str, Dict[str, str]],
    fallback: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        if chunk_id and chunk_id in chunk_lookup:
            chunks.append(chunk_lookup[chunk_id])
    if not chunks:
        chunks = fallback
    return chunks

def _filter_low_info_chunks(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [chunk for chunk in chunks if not _is_low_info_chunk(chunk)]


def _select_evidence_chunks(
    concept: Concept,
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
) -> List[Dict[str, str]]:
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    fallback = search_index(
        concept.get("name", ""),
        chunks,
        embeddings,
        lambda text: client.embed(settings.embed_model, text),
        top_k=8,
    )
    evidence_chunks = _collect_evidence_by_citations(
        concept.get("citations", []),
        chunk_lookup,
        fallback,
    )
    evidence_chunks = _filter_low_info_chunks(evidence_chunks)
    if not evidence_chunks:
        evidence_chunks = _filter_low_info_chunks(fallback)
    if not evidence_chunks:
        evidence_chunks = _filter_low_info_chunks(chunks)[:3]

    unique = []
    seen = set()
    for chunk in evidence_chunks:
        if chunk["chunk_id"] in seen:
            continue
        seen.add(chunk["chunk_id"])
        unique.append(chunk)
    return unique[:6]


def generate_question_for_concept(
    client: OllamaClient,
    settings: Settings,
    concept: Concept,
    evidence_chunks: List[Dict[str, str]],
    question_id: str,
    question_type: str,
) -> Question:
    attempt = 0
    question: Question = {}
    evidence_text = _format_evidence(
        evidence_chunks,
        settings.evidence_budget_tokens,
        settings.max_input_chars,
    )
    while attempt <= settings.question_retries:
        prompt = QUESTION_PROMPT.format(
            question_id=question_id,
            question_type=question_type,
            concept=json.dumps(concept, ensure_ascii=False),
            evidence=evidence_text,
        )
        try:
            data = client.chat_json(
                settings.chat_model,
                [{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                timeout=settings.chat_timeout,
            )
        except Exception as exc:
            print(f"Question {question_id} generation error: {exc}")
            data = {}
        if isinstance(data, dict) and data.get("insufficient_evidence"):
            attempt += 1
            continue
        question = normalize_question(data if isinstance(data, dict) else {}, question_id)
        question["id"] = question_id
        question["type"] = question_type
        question["evidence_quotes"] = []
        if not question.get("concept_tags"):
            question["concept_tags"] = [concept.get("name", "concept")]
        if question_type == "tf":
            fixed = _fix_tf_question(question.get("question", ""))
            if fixed:
                question["question"] = fixed
            else:
                print(f"Question {question_id} rejected (tf question format).")
                attempt += 1
                continue
        if question_type == "short" and question.get("answer", "").lower() in {"true", "false"}:
            print(f"Question {question_id} rejected (short answer format).")
            attempt += 1
            continue
        if question_type == "mcq" and _has_banned_mcq_choice(question.get("choices", [])):
            print(f"Question {question_id} rejected (banned MCQ choice).")
            attempt += 1
            continue
        if _is_meta_question(question.get("question", "")):
            print(f"Question {question_id} rejected (meta question).")
            attempt += 1
            continue
        _ensure_citations(question, evidence_chunks)
        _ensure_evidence_quotes(question, evidence_chunks)
        _ensure_rationale_quotes(question)
        if _validate_question(question, settings):
            return question
        print(f"Question {question_id} missing citations or fields; retry {attempt + 1}.")
        attempt += 1
    return question


def verify_questions(
    client: OllamaClient,
    settings: Settings,
    questions: List[Question],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    concept_lookup: Dict[str, Concept],
) -> List[Question]:
    log_step("Verify questions")
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    verified: List[Question] = []

    for question in questions:
        attempts = 0
        current = question
        while attempts <= settings.verify_retries:
            fallback = search_index(
                current.get("question", ""),
                chunks,
                embeddings,
                lambda text: client.embed(settings.embed_model, text),
                top_k=5,
            )
            fallback = _filter_low_info_chunks(fallback)
            evidence_chunks = _collect_evidence_by_citations(
                current.get("citations", []),
                chunk_lookup,
                fallback,
            )
            evidence_chunks = _filter_low_info_chunks(evidence_chunks)
            if not evidence_chunks:
                evidence_chunks = fallback
            if not evidence_chunks:
                evidence_chunks = _filter_low_info_chunks(chunks)[:3]
            evidence_text = _format_evidence(
                evidence_chunks,
                settings.evidence_budget_tokens,
                settings.max_input_chars,
            )
            prompt = VERIFY_PROMPT.format(
                question_json=json.dumps(current, ensure_ascii=False),
                evidence=evidence_text,
            )
            try:
                data = client.chat_json(
                    settings.chat_model,
                    [{"role": "user", "content": prompt}],
                    options={"temperature": 0.2},
                    timeout=settings.chat_timeout,
                )
            except Exception as exc:
                print(f"Verify question {current.get('id')} error: {exc}")
                data = {}

            supported = bool(data.get("supported"))
            evidence_full_text = "\n".join(chunk["text"] for chunk in evidence_chunks)
            if not current.get("citations"):
                supported = False
            if _is_meta_question(current.get("question", "")):
                supported = False
            if current.get("type") == "tf" and _is_question_sentence(current.get("question", "")):
                supported = False
            if current.get("type") == "short" and current.get("answer", "").lower() in {"true", "false"}:
                supported = False
            if current.get("type") == "mcq" and _has_banned_mcq_choice(current.get("choices", [])):
                supported = False
            if _contains_external_reference(
                f"{current.get('question', '')} {current.get('answer', '')} {current.get('rationale', '')}",
                evidence_full_text,
            ):
                supported = False
            for quote in current.get("evidence_quotes", []):
                chunk = chunk_lookup.get(quote.get("chunk_id"))
                if not chunk or quote.get("quote") not in chunk["text"]:
                    supported = False
                    break

            if supported:
                break

            revised = data.get("revised_question")
            if isinstance(revised, dict):
                print(f"Question {current.get('id')} rewritten.")
                current = normalize_question(revised, current.get("id", ""))
                current["evidence_quotes"] = []
                if current.get("type") == "tf":
                    fixed = _fix_tf_question(current.get("question", ""))
                    if fixed:
                        current["question"] = fixed
                    else:
                        attempts += 1
                        continue
                if current.get("type") == "short" and current.get("answer", "").lower() in {"true", "false"}:
                    attempts += 1
                    continue
                if current.get("type") == "mcq" and _has_banned_mcq_choice(current.get("choices", [])):
                    attempts += 1
                    continue
                _ensure_citations(current, evidence_chunks)
                _ensure_evidence_quotes(current, evidence_chunks)
                _ensure_rationale_quotes(current)
                if _validate_question(current, settings):
                    break
            attempts += 1

        if not _validate_question(current, settings):
            concept_name = current.get("concept_tags", [""])[0]
            concept = concept_lookup.get(concept_name)
            if concept:
                evidence_chunks = _select_evidence_chunks(concept, chunks, embeddings, client, settings)
                regenerated = generate_question_for_concept(
                    client,
                    settings,
                    concept,
                    evidence_chunks,
                    current.get("id", "q"),
                    current.get("type", settings.question_types[0]),
                )
                if _validate_question(regenerated, settings):
                    current = regenerated
        if not _validate_question(current, settings):
            print(f"Question {current.get('id')} dropped (invalid).")
            continue
        verified.append(current)

    return verified


def run_pipeline(pdf_path: str, settings: Settings) -> QuizOutput:
    random.seed(settings.seed)
    pages = ingest_pdf(pdf_path, settings)
    if not pages:
        raise ValueError("No pages found in PDF.")

    pages = _filter_pages(pages, settings)

    chunk_tokens = settings.chunk_tokens
    overlap_tokens = settings.overlap_tokens
    if settings.chunk_chars and settings.chunk_tokens <= 0:
        chunk_tokens = max(200, int(settings.chunk_chars * 0.45))
    if settings.overlap_chars and settings.overlap_tokens <= 0:
        overlap_tokens = max(20, int(settings.overlap_chars * 0.45))

    chunks = chunk_pages(pages, chunk_tokens, overlap_tokens, settings.min_chunk_tokens)
    if not chunks:
        raise ValueError("No text chunks generated from PDF.")

    filtered_chunks = [chunk for chunk in chunks if not _is_low_info_chunk(chunk)]
    if filtered_chunks:
        removed = len(chunks) - len(filtered_chunks)
        if removed > 0:
            print(f"Filtered {removed} low-info chunks.")
        chunks = filtered_chunks

    client = OllamaClient(settings.base_url, default_timeout=settings.chat_timeout)
    if not client.check_health():
        raise RuntimeError(f"Cannot reach Ollama at {settings.base_url}")

    embeddings = _build_or_load_embeddings(pdf_path, chunks, client, settings)
    selected_chunks = _select_chunk_set(pages, chunks, embeddings, client, settings)
    print(f"Selected {len(selected_chunks)} chunks for map-reduce.")

    mini_summaries = map_summarize(client, settings, selected_chunks)
    summary_block = reduce_summarize(client, settings, mini_summaries, selected_chunks, chunks, embeddings)

    keypoints = summary_block["keypoints"]
    print(f"Keypoints: {len(keypoints)}")

    concepts = extract_concepts(client, settings, keypoints, mini_summaries)
    concept_lookup = {concept["name"]: concept for concept in concepts}

    questions: List[Question] = []
    for idx, concept in enumerate(concepts[: settings.question_count], start=1):
        evidence_chunks = _select_evidence_chunks(concept, chunks, embeddings, client, settings)
        question_type = settings.question_types[(idx - 1) % len(settings.question_types)]
        question = generate_question_for_concept(
            client,
            settings,
            concept,
            evidence_chunks,
            f"q{idx}",
            question_type,
        )
        questions.append(question)

    print(f"Generated questions: {len(questions)}")

    verified_questions = verify_questions(
        client,
        settings,
        questions,
        chunks,
        embeddings,
        concept_lookup,
    )

    if len(verified_questions) < settings.question_count and concepts:
        attempts = 0
        next_id = len(verified_questions) + 1
        max_attempts = settings.question_count * (settings.question_retries + 2)
        while len(verified_questions) < settings.question_count and attempts < max_attempts:
            concept = concepts[attempts % len(concepts)]
            evidence_chunks = _select_evidence_chunks(concept, chunks, embeddings, client, settings)
            question_type = settings.question_types[(next_id - 1) % len(settings.question_types)]
            candidate = generate_question_for_concept(
                client,
                settings,
                concept,
                evidence_chunks,
                f"q{next_id}",
                question_type,
            )
            verified = verify_questions(
                client,
                settings,
                [candidate],
                chunks,
                embeddings,
                concept_lookup,
            )
            if verified:
                verified_questions.append(verified[0])
                next_id += 1
            attempts += 1

    return {
        "summary": summary_block,
        "questions": verified_questions,
    }
