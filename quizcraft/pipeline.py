import json
from typing import Dict, List

from quizcraft.chunker import chunk_pages
from quizcraft.config import Settings
from quizcraft.ingest import ingest_pdf
from quizcraft.ollama_client import OllamaClient
from quizcraft.prompts import QUESTION_PROMPT, SUMMARY_PROMPT, VERIFY_PROMPT
from quizcraft.retrieval import build_index, search_index
from quizcraft.schemas import Question, QuizOutput, normalize_question, normalize_questions
from quizcraft.utils import log_step


def _build_context(chunks: List[Dict[str, str]], max_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        text = chunk["text"]
        if total + len(text) + 1 > max_chars:
            break
        parts.append(text)
        total += len(text) + 1
    return "\n".join(parts)


def _format_evidence(chunks: List[Dict[str, str]]) -> str:
    evidence = [
        {"chunk_id": c["chunk_id"], "page": c["page"], "text": c["text"]}
        for c in chunks
    ]
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _format_keypoints(keypoints: List[str]) -> str:
    return "\n".join(f"- {kp}" for kp in keypoints)


def generate_summary(client: OllamaClient, settings: Settings, chunks: List[Dict[str, str]]) -> Dict[str, List[str] | str]:
    log_step("Summarize")
    context = _build_context(chunks, settings.max_context_chars)
    prompt = SUMMARY_PROMPT.format(
        summary_min=settings.summary_min_chars,
        summary_max=settings.summary_max_chars,
        context=context,
    )
    data = client.chat_json(
        settings.chat_model,
        [{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    summary = str(data.get("summary", "")).strip()
    keypoints = data.get("keypoints") if isinstance(data.get("keypoints"), list) else []
    keypoints = [str(k).strip() for k in keypoints if str(k).strip()]
    return {"summary": summary, "keypoints": keypoints}


def _collect_evidence(
    keypoints: List[str],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    client: OllamaClient,
    settings: Settings,
) -> List[Dict[str, str]]:
    log_step("Collect evidence")
    collected: Dict[str, Dict[str, str]] = {}
    for kp in keypoints:
        matches = search_index(
            kp,
            chunks,
            embeddings,
            lambda text: client.embed(settings.embed_model, text),
            top_k=settings.evidence_per_keypoint,
        )
        for match in matches:
            collected[match["chunk_id"]] = match
    evidence = list(collected.values())
    print(f"Evidence chunks: {len(evidence)}")
    return evidence


def generate_questions(
    client: OllamaClient,
    settings: Settings,
    keypoints: List[str],
    evidence: List[Dict[str, str]],
) -> List[Question]:
    log_step("Generate questions")
    prompt = QUESTION_PROMPT.format(
        keypoints=_format_keypoints(keypoints),
        evidence=_format_evidence(evidence),
        question_count=settings.question_count,
        question_types=", ".join(settings.question_types),
    )
    data = client.chat_json(
        settings.chat_model,
        [{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )
    questions = normalize_questions(data.get("questions"))
    return questions


def _collect_citation_chunks(
    question: Question,
    chunk_lookup: Dict[str, Dict[str, str]],
    fallback_chunks: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for citation in question.get("citations", []):
        chunk_id = citation.get("chunk_id")
        if chunk_id and chunk_id in chunk_lookup:
            chunks.append(chunk_lookup[chunk_id])
    if not chunks:
        chunks = fallback_chunks
    return chunks


def verify_questions(
    client: OllamaClient,
    settings: Settings,
    questions: List[Question],
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
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
            evidence_chunks = _collect_citation_chunks(current, chunk_lookup, fallback)
            prompt = VERIFY_PROMPT.format(
                question_json=json.dumps(current, ensure_ascii=False),
                evidence=_format_evidence(evidence_chunks),
            )
            data = client.chat_json(
                settings.chat_model,
                [{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
            )
            supported = bool(data.get("supported"))
            if supported:
                break

            revised = data.get("revised_question")
            if isinstance(revised, dict):
                print(f"Question {current.get('id')} rewritten.")
                current = normalize_question(revised, current.get("id", ""))
            else:
                break

            attempts += 1

        verified.append(current)

    return verified


def run_pipeline(pdf_path: str, settings: Settings) -> QuizOutput:
    pages = ingest_pdf(pdf_path)
    if not pages:
        raise ValueError("No pages found in PDF.")

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks generated from PDF.")

    client = OllamaClient(settings.base_url)
    if not client.check_health():
        raise RuntimeError(f"Cannot reach Ollama at {settings.base_url}")

    embeddings = build_index(chunks, lambda text: client.embed(settings.embed_model, text))
    summary_data = generate_summary(client, settings, chunks)
    evidence = _collect_evidence(summary_data["keypoints"], chunks, embeddings, client, settings)
    questions = generate_questions(client, settings, summary_data["keypoints"], evidence)
    verified = verify_questions(client, settings, questions, chunks, embeddings)

    return {
        "summary": summary_data["summary"],
        "keypoints": summary_data["keypoints"],
        "questions": verified,
    }
