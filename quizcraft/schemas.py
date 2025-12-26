from typing import Any, Dict, List, TypedDict


class Citation(TypedDict):
    page: int
    chunk_id: str


class Question(TypedDict, total=False):
    id: str
    type: str
    question: str
    choices: List[str]
    answer: str
    rationale: str
    citations: List[Citation]
    difficulty: str
    concept_tags: List[str]


class QuizOutput(TypedDict):
    summary: str
    keypoints: List[str]
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


def normalize_question(raw: Dict[str, Any], fallback_id: str) -> Question:
    q_type = str(raw.get("type", "tf")).lower()
    if q_type not in {"tf", "mcq", "short"}:
        q_type = "tf"

    choices = raw.get("choices") if q_type == "mcq" else None
    if q_type == "mcq" and not isinstance(choices, list):
        choices = []

    question: Question = {
        "id": str(raw.get("id") or fallback_id),
        "type": q_type,
        "question": str(raw.get("question", "")).strip(),
        "answer": str(raw.get("answer", "")).strip(),
        "rationale": str(raw.get("rationale", "")).strip(),
        "citations": _normalize_citations(raw.get("citations")),
        "difficulty": str(raw.get("difficulty", "medium")).lower(),
        "concept_tags": raw.get("concept_tags") if isinstance(raw.get("concept_tags"), list) else [],
    }

    if q_type == "mcq":
        question["choices"] = [str(c).strip() for c in choices if str(c).strip()]

    return question


def normalize_questions(raw_questions: Any) -> List[Question]:
    questions: List[Question] = []
    if not isinstance(raw_questions, list):
        return questions
    for idx, raw in enumerate(raw_questions, start=1):
        if isinstance(raw, dict):
            questions.append(normalize_question(raw, f"q{idx}"))
    return questions
