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

def _choice_text(choice) -> str:
    if isinstance(choice, dict):
        for key in ("text", "label", "content", "choice", "value"):
            if choice.get(key):
                return str(choice[key]).strip()
    return str(choice).strip()


def _format_choice_line(choice: str, letter: str) -> str:
    text = _choice_text(choice)
    if re.match(rf"^{letter}[\s\).、:-]", text):
        return text
    return f"{letter}. {text}"



def _extract_choice_letter(text: str) -> str:
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()
    return ""


def _grade_tf(user_answer: str, correct: str) -> bool:
    return _normalize_tf(user_answer) == _normalize_tf(correct)


def _grade_mcq(user_answer: str, correct: str, choices: List[str], correct_option: str = "") -> bool:
    correct_letter = correct_option.upper() if correct_option else _extract_choice_letter(correct)
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


def _grade_short(user_answer: str, correct: str) -> bool:
    return user_answer.strip().lower() == correct.strip().lower()


def _grade_calc(user_answer: str, correct: str) -> bool:
    return re.sub(r"\s+", "", user_answer) == re.sub(r"\s+", "", correct)


def grade_answer(question: Question, user_answer: str) -> bool:
    q_type = question.get("type", "tf")
    if q_type == "mcq":
        return _grade_mcq(
            user_answer,
            question.get("answer", ""),
            question.get("choices", []),
            question.get("correct_option", ""),
        )
    if q_type == "short":
        return _grade_short(user_answer, question.get("answer", ""))
    if q_type == "calc":
        return _grade_calc(user_answer, question.get("final_answer", question.get("answer", "")))
    return _grade_tf(user_answer, question.get("answer", ""))


def _format_citations(question: Question) -> str:
    citations = question.get("citations", [])
    if not citations:
        return "N/A"
    return ", ".join(f"p{c['page']}:{c['chunk_id']}" for c in citations)


def _format_quotes(question: Question) -> str:
    quotes = question.get("evidence_quotes", [])
    if not quotes:
        return "N/A"
    return " / ".join(f"{q['quote']}" for q in quotes)


def run_quiz(questions: List[Question]) -> Tuple[int, List[str]]:
    print("\n=== Quiz Mode ===")
    score = 0
    wrong_ids: List[str] = []

    for idx, question in enumerate(questions, start=1):
        print(f"\nQ{idx}. ({question.get('type')}) {question.get('question')}")
        if question.get("type") == "mcq":
            for i, choice in enumerate(question.get("choices", []), start=1):
                letter = chr(ord("A") + i - 1)
                formatted = _format_choice_line(choice, letter)
                print(f"  {formatted}")

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
        if question.get("type") == "calc":
            print("Step-by-step:", " / ".join(question.get("step_by_step", [])))
            print("Final Answer:", question.get("final_answer", ""))
        print("Rationale:", question.get("rationale", ""))
        print("Citations:", _format_citations(question))
        print("Evidence Quotes:", _format_quotes(question))

    print("\n=== Quiz Summary ===")
    print(f"Score: {score}/{len(questions)}")
    if wrong_ids:
        print("Incorrect questions:", ", ".join(wrong_ids))

    return score, wrong_ids
