import json
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
    MiniSummary,
    Question,
    QuizOutput,
    normalize_concepts,
    normalize_mini_summary,
    normalize_question,
)
from quizcraft.utils import log_step

_CITATION_RE = re.compile(r"p\d+_c\d+")


def _limit_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


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


def _format_evidence(chunks: List[Dict[str, str]], max_chars: int) -> str:
    entries = []
    total = 0
    for chunk in chunks:
        entry = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
        }
        serialized = json.dumps(entry, ensure_ascii=False)
        if total + len(serialized) > max_chars and entries:
            break
        entries.append(entry)
        total += len(serialized)
        if total >= max_chars:
            break
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _format_mini_summaries(mini_summaries: List[MiniSummary], max_chars: int) -> str:
    entries = []
    total = 0
    for summary in mini_summaries:
        entry = {
            "mini_summary": summary["mini_summary"],
            "keywords": summary["keywords"],
            "citations": summary["citations"],
        }
        serialized = json.dumps(entry, ensure_ascii=False)
        if total + len(serialized) > max_chars and entries:
            break
        entries.append(entry)
        total += len(serialized)
        if total >= max_chars:
            break
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _citation_tag(citation: Dict[str, str | int]) -> str:
    return f"p{citation['page']}:{citation['chunk_id']}"


def _attach_citations_to_summary(summary: str, citations: List[Dict[str, str | int]], settings: Settings) -> str:
    if _has_citation(summary) or not citations:
        return summary
    tags = [_citation_tag(c) for c in citations][:3]
    citation_text = f"（參考 {', '.join(tags)}）"
    if len(summary) + len(citation_text) > settings.summary_max_chars:
        summary = summary[: max(0, settings.summary_max_chars - len(citation_text))]
    return f"{summary}{citation_text}"


def _attach_citations_to_keypoints(
    keypoints: List[str],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
) -> List[str]:
    updated: List[str] = []
    for kp in keypoints:
        if _has_citation(kp):
            updated.append(kp)
            continue
        matches = search_index(
            kp,
            chunks,
            embeddings,
            lambda text: client.embed(settings.embed_model, text),
            top_k=1,
        )
        if matches:
            citation = {"page": matches[0]["page"], "chunk_id": matches[0]["chunk_id"]}
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
    if not question.get("citations"):
        return False
    if q_type == "mcq":
        choices = question.get("choices", [])
        if not isinstance(choices, list) or len(choices) < 4:
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


def map_summarize(
    client: OllamaClient,
    settings: Settings,
    chunks: List[Dict[str, str]],
) -> List[MiniSummary]:
    log_step("Map summarize")
    summaries: List[MiniSummary] = []
    for chunk in chunks:
        chunk_text = _limit_text(chunk["text"], settings.chunk_chars)
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
            mini["mini_summary"] = chunk_text[:120]
        summaries.append(mini)
    print(f"Mini summaries: {len(summaries)}")
    return summaries


def reduce_summarize(
    client: OllamaClient,
    settings: Settings,
    mini_summaries: List[MiniSummary],
) -> Tuple[str, List[str], List[Dict[str, str | int]]]:
    log_step("Reduce summarize")
    context = _format_mini_summaries(mini_summaries, settings.max_input_chars)
    prompt = REDUCE_SUMMARY_PROMPT.format(
        summary_min=settings.summary_min_chars,
        summary_max=settings.summary_max_chars,
        mini_summaries=context,
    )
    attempts = 0
    summary = ""
    keypoints: List[str] = []
    citations: List[Dict[str, str | int]] = []

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

        summary = str(data.get("summary", "")).strip()
        raw_keypoints = data.get("keypoints") if isinstance(data.get("keypoints"), list) else []
        keypoints = [str(k).strip() for k in raw_keypoints if str(k).strip()]
        citations = _normalize_citations(data.get("citations"))
        if not citations:
            citations = [ms["citations"][0] for ms in mini_summaries if ms.get("citations")][:3]
        if _validate_summary(summary, keypoints, settings):
            break
        attempts += 1

    if not _validate_summary(summary, keypoints, settings):
        print("Reduce summary fallback triggered.")
        summary, keypoints, citations = _fallback_summary_from_mini(mini_summaries, settings)

    return summary, keypoints, citations


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
    context = _format_mini_summaries(mini_summaries, settings.max_input_chars)
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


def generate_question_for_concept(
    client: OllamaClient,
    settings: Settings,
    concept: Concept,
    evidence_chunks: List[Dict[str, str]],
    question_id: str,
) -> Question:
    attempt = 0
    question: Question = {}
    evidence_text = _format_evidence(evidence_chunks, settings.max_input_chars)
    while attempt <= settings.question_retries:
        prompt = QUESTION_PROMPT.format(
            question_types=", ".join(settings.question_types),
            question_id=question_id,
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
        question = normalize_question(data, question_id)
        question["id"] = question_id
        if not question.get("concept_tags"):
            question["concept_tags"] = [concept.get("name", "concept")]
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
                top_k=3,
            )
            evidence_chunks = _collect_evidence_by_citations(
                current.get("citations", []),
                chunk_lookup,
                fallback,
            )
            prompt = VERIFY_PROMPT.format(
                question_json=json.dumps(current, ensure_ascii=False),
                evidence=_format_evidence(evidence_chunks, settings.max_input_chars),
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
            if not current.get("citations"):
                supported = False

            if supported:
                break

            revised = data.get("revised_question")
            if isinstance(revised, dict):
                print(f"Question {current.get('id')} rewritten.")
                current = normalize_question(revised, current.get("id", ""))
                if _validate_question(current, settings):
                    break
            attempts += 1

        if not _validate_question(current, settings):
            concept_name = current.get("concept_tags", [""])[0]
            concept = concept_lookup.get(concept_name)
            if concept:
                fallback = search_index(
                    concept.get("name", ""),
                    chunks,
                    embeddings,
                    lambda text: client.embed(settings.embed_model, text),
                    top_k=3,
                )
                evidence_chunks = _collect_evidence_by_citations(
                    concept.get("citations", []),
                    chunk_lookup,
                    fallback,
                )
                regenerated = generate_question_for_concept(
                    client,
                    settings,
                    concept,
                    evidence_chunks,
                    current.get("id", "q"),
                )
                if _validate_question(regenerated, settings):
                    current = regenerated
        verified.append(current)

    return verified


def run_pipeline(pdf_path: str, settings: Settings) -> QuizOutput:
    random.seed(settings.seed)
    pages = ingest_pdf(pdf_path)
    if not pages:
        raise ValueError("No pages found in PDF.")

    chunks = chunk_pages(pages, settings.chunk_chars, settings.overlap_chars)
    if not chunks:
        raise ValueError("No text chunks generated from PDF.")

    client = OllamaClient(settings.base_url, default_timeout=settings.chat_timeout)
    if not client.check_health():
        raise RuntimeError(f"Cannot reach Ollama at {settings.base_url}")

    embeddings = build_index(chunks, lambda text: client.embed(settings.embed_model, text))
    selected_chunks = _select_chunk_set(pages, chunks, embeddings, client, settings)
    print(f"Selected {len(selected_chunks)} chunks for map-reduce.")

    mini_summaries = map_summarize(client, settings, selected_chunks)
    summary, keypoints, summary_citations = reduce_summarize(client, settings, mini_summaries)

    summary_with_citations = _attach_citations_to_summary(summary, summary_citations, settings)
    keypoints_with_citations = _attach_citations_to_keypoints(
        keypoints,
        chunks,
        embeddings,
        client,
        settings,
    )
    print(f"Keypoints: {len(keypoints_with_citations)}")

    concepts = extract_concepts(client, settings, keypoints_with_citations, mini_summaries)
    concept_lookup = {concept["name"]: concept for concept in concepts}

    questions: List[Question] = []
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    for idx, concept in enumerate(concepts[: settings.question_count], start=1):
        fallback = search_index(
            concept.get("name", ""),
            chunks,
            embeddings,
            lambda text: client.embed(settings.embed_model, text),
            top_k=3,
        )
        evidence_chunks = _collect_evidence_by_citations(
            concept.get("citations", []),
            chunk_lookup,
            fallback,
        )
        if not evidence_chunks:
            evidence_chunks = selected_chunks[:1]
        question = generate_question_for_concept(
            client,
            settings,
            concept,
            evidence_chunks,
            f"q{idx}",
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

    return {
        "summary": summary_with_citations,
        "keypoints": keypoints_with_citations,
        "questions": verified_questions,
    }
