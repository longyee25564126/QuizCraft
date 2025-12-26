import re
from typing import List, Tuple

from quizcraft.schemas import Question


def _normalize_tf(text: str) -> str:
    normalized = text.strip().lower()
    if normalized in {"t", "true", "yes", "y", "對", "正确", "正確"}:
        return "true"
    if normalized in {"f", "false", "no", "n", "錯", "错误", "錯誤"}:
        return "false"
    return normalized


def _extract_choice_letter(text: str) -> str:
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()
    return ""


def _grade_tf(user_answer: str, correct: str) -> bool:
    return _normalize_tf(user_answer) == _normalize_tf(correct)


def _grade_mcq(user_answer: str, correct: str, choices: List[str]) -> bool:
    correct_letter = _extract_choice_letter(correct)
    if not correct_letter and choices:
        for idx, choice in enumerate(choices):
            if correct.strip().lower() == choice.strip().lower():
                correct_letter = chr(ord("A") + idx)
                break

    user_letter = _extract_choice_letter(user_answer)
    if user_letter:
        return user_letter == correct_letter

    if choices:
        for idx, choice in enumerate(choices):
            if user_answer.strip().lower() == choice.strip().lower():
                return chr(ord("A") + idx) == correct_letter

    return False


def grade_answer(question: Question, user_answer: str) -> bool:
    q_type = question.get("type", "tf")
    if q_type == "mcq":
        return _grade_mcq(user_answer, question.get("answer", ""), question.get("choices", []))
    if q_type == "short":
        return user_answer.strip().lower() == question.get("answer", "").strip().lower()
    return _grade_tf(user_answer, question.get("answer", ""))


def _format_citations(question: Question) -> str:
    citations = question.get("citations", [])
    if not citations:
        return "N/A"
    return ", ".join(f"p{c['page']}:{c['chunk_id']}" for c in citations)


def run_quiz(questions: List[Question]) -> Tuple[int, List[str]]:
    print("\n=== Quiz Mode ===")
    score = 0
    wrong_ids: List[str] = []

    for idx, question in enumerate(questions, start=1):
        print(f"\nQ{idx}. ({question.get('type')}) {question.get('question')}")
        if question.get("type") == "mcq":
            for i, choice in enumerate(question.get("choices", []), start=1):
                letter = chr(ord("A") + i - 1)
                print(f"  {letter}. {choice}")

        user_answer = input("Your answer: ").strip()
        is_correct = grade_answer(question, user_answer)

        if not is_correct:
            hint = question.get("rationale", "")
            if hint:
                print(f"Hint: {hint[:60]}...")
            retry_answer = input("Try again (or press Enter to skip): ").strip()
            if retry_answer:
                is_correct = grade_answer(question, retry_answer)

        if is_correct:
            print("✅ Correct!")
            score += 1
        else:
            print("❌ Incorrect.")
            wrong_ids.append(question.get("id", f"q{idx}"))

        print("Answer:", question.get("answer", ""))
        print("Rationale:", question.get("rationale", ""))
        print("Citations:", _format_citations(question))

    print("\n=== Quiz Summary ===")
    print(f"Score: {score}/{len(questions)}")
    if wrong_ids:
        print("Incorrect questions:", ", ".join(wrong_ids))

    return score, wrong_ids
