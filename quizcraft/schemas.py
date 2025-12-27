import re
from typing import Any, Dict, List, TypedDict


class Citation(TypedDict):
    page: int
    chunk_id: str


class EvidenceQuote(TypedDict):
    page: int
    chunk_id: str
    quote: str


class MiniSummary(TypedDict):
    page: int
    chunk_id: str
    mini_summary: str
    keywords: List[str]
    citations: List[Citation]
    evidence_quotes: List[EvidenceQuote]


class Concept(TypedDict):
    name: str
    description: str
    citations: List[Citation]
    difficulty: str


class SummarySection(TypedDict):
    title: str
    summary: str
    citations: List[Citation]


class SummaryBlock(TypedDict):
    overview: str
    sections: List[SummarySection]
    keypoints: List[str]


class Question(TypedDict, total=False):
    id: str
    type: str
    question: str
    choices: List[str]
    answer: str
    correct_option: str
    rationale: str
    citations: List[Citation]
    evidence_quotes: List[EvidenceQuote]
    difficulty: str
    concept_tags: List[str]
    step_by_step: List[str]
    final_answer: str


class QuizOutput(TypedDict):
    summary: SummaryBlock
    questions: List[Question]


def _normalize_citations(raw: Any) -> List[Citation]:
    citations: List[Citation] = []
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


def _normalize_evidence_quotes(raw: Any) -> List[EvidenceQuote]:
    quotes: List[EvidenceQuote] = []
    if not isinstance(raw, list):
        return quotes
    for item in raw:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        chunk_id = item.get("chunk_id") or item.get("chunkId")
        quote = str(item.get("quote", "")).strip()
        if page is None or chunk_id is None or not quote:
            continue
        try:
            page_num = int(page)
        except (ValueError, TypeError):
            continue
        quotes.append({"page": page_num, "chunk_id": str(chunk_id), "quote": quote})
    return quotes


def _strip_choice_prefix(text: str) -> str:
    text = text.strip()
    return re.sub(r"^[A-Da-d][\s\).、:-]+", "", text).strip()


def _choice_text(choice: Any) -> str:
    if isinstance(choice, dict):
        for key in ("text", "label", "content", "choice", "value"):
            if key in choice and choice[key]:
                return str(choice[key]).strip()
    return str(choice).strip()


def _normalize_mcq_choices(choices: List[Any]) -> List[str]:
    normalized: List[str] = []
    for idx in range(4):
        if idx >= len(choices):
            break
        raw = _choice_text(choices[idx])
        raw = _strip_choice_prefix(raw)
        prefix = f"{chr(ord('A') + idx)} "
        normalized.append(f"{prefix}{raw}".strip())
    return normalized


def normalize_question(raw: Dict[str, Any], fallback_id: str) -> Question:
    q_type = str(raw.get("type", "tf")).lower()
    if q_type not in {"tf", "mcq", "short", "calc"}:
        q_type = "tf"

    choices_raw = raw.get("choices") if q_type == "mcq" else None
    if q_type == "mcq" and not isinstance(choices_raw, list):
        choices_raw = []

    question: Question = {
        "id": str(raw.get("id") or fallback_id),
        "type": q_type,
        "question": str(raw.get("question", "")).strip(),
        "answer": str(raw.get("answer", "")).strip(),
        "rationale": str(raw.get("rationale", "")).strip(),
        "citations": _normalize_citations(raw.get("citations")),
        "evidence_quotes": _normalize_evidence_quotes(raw.get("evidence_quotes")),
        "difficulty": str(raw.get("difficulty", "medium")).lower(),
        "concept_tags": raw.get("concept_tags") if isinstance(raw.get("concept_tags"), list) else [],
    }

    if q_type == "mcq":
        choices = _normalize_mcq_choices(choices_raw or [])
        question["choices"] = choices

        correct = str(raw.get("correct_option") or raw.get("answer", "")).strip().upper()
        if correct and correct[0] in {"A", "B", "C", "D"}:
            question["correct_option"] = correct[0]
        else:
            answer_text = _strip_choice_prefix(str(raw.get("answer", "")))
            for idx, choice in enumerate(choices):
                if answer_text and answer_text.lower() in choice.lower():
                    question["correct_option"] = chr(ord("A") + idx)
                    break
        if question.get("correct_option"):
            question["answer"] = question["correct_option"]

    elif q_type == "tf":
        answer = question["answer"].lower()
        if answer in {"t", "true", "yes", "y", "對", "正确", "正確"}:
            question["answer"] = "true"
        elif answer in {"f", "false", "no", "n", "錯", "错误", "錯誤"}:
            question["answer"] = "false"

    elif q_type == "calc":
        step_by_step = raw.get("step_by_step")
        if isinstance(step_by_step, list):
            question["step_by_step"] = [str(step).strip() for step in step_by_step if str(step).strip()]
        question["final_answer"] = str(raw.get("final_answer", raw.get("answer", ""))).strip()

    return question


def normalize_questions(raw_questions: Any) -> List[Question]:
    questions: List[Question] = []
    if not isinstance(raw_questions, list):
        return questions
    for idx, raw in enumerate(raw_questions, start=1):
        if isinstance(raw, dict):
            questions.append(normalize_question(raw, f"q{idx}"))
    return questions


def normalize_mini_summary(raw: Dict[str, Any], page: int, chunk_id: str) -> MiniSummary:
    mini_summary = str(raw.get("mini_summary", "")).strip()
    keywords_raw = raw.get("keywords") if isinstance(raw.get("keywords"), list) else []
    keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
    citations = _normalize_citations(raw.get("citations"))
    if not citations:
        citations = [{"page": page, "chunk_id": chunk_id}]
    quotes = _normalize_evidence_quotes(raw.get("evidence_quotes"))
    return {
        "page": page,
        "chunk_id": chunk_id,
        "mini_summary": mini_summary,
        "keywords": keywords,
        "citations": citations,
        "evidence_quotes": quotes,
    }


def normalize_concepts(raw: Any) -> List[Concept]:
    concepts: List[Concept] = []
    if not isinstance(raw, list):
        return concepts
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        citations = _normalize_citations(item.get("citations"))
        difficulty = str(item.get("difficulty", "medium")).lower()
        if name:
            concepts.append(
                {
                    "name": name,
                    "description": description,
                    "citations": citations,
                    "difficulty": difficulty,
                }
            )
    return concepts
