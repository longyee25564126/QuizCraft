import json
from typing import Dict

from quizcraft.schemas import QuizOutput


def _format_citations(citations) -> str:
    if not citations:
        return "N/A"
    return ", ".join(f"p{c['page']}:{c['chunk_id']}" for c in citations)


def export_markdown(output: QuizOutput, path: str) -> None:
    lines = []
    lines.append("# QuizCraft Output\n")
    lines.append("## Summary\n")
    lines.append(output["summary"] + "\n")

    lines.append("## Keypoints\n")
    for kp in output["keypoints"]:
        lines.append(f"- {kp}")
    lines.append("")

    lines.append("## Questions\n")
    for idx, q in enumerate(output["questions"], start=1):
        lines.append(f"### Q{idx}: {q.get('question', '')}")
        if q.get("type") == "mcq":
            for i, choice in enumerate(q.get("choices", []), start=1):
                letter = chr(ord("A") + i - 1)
                lines.append(f"- {letter}. {choice}")
        lines.append(f"**Answer:** {q.get('answer', '')}")
        lines.append(f"**Rationale:** {q.get('rationale', '')}")
        lines.append(f"**Citations:** {_format_citations(q.get('citations', []))}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def export_text(output: QuizOutput, path: str) -> None:
    lines = []
    lines.append("QuizCraft Output\n")
    lines.append("Summary:\n")
    lines.append(output["summary"] + "\n")

    lines.append("Keypoints:\n")
    for kp in output["keypoints"]:
        lines.append(f"- {kp}")
    lines.append("")

    lines.append("Questions:\n")
    for idx, q in enumerate(output["questions"], start=1):
        lines.append(f"Q{idx}: {q.get('question', '')}")
        if q.get("type") == "mcq":
            for i, choice in enumerate(q.get("choices", []), start=1):
                letter = chr(ord("A") + i - 1)
                lines.append(f"  {letter}. {choice}")
        lines.append(f"Answer: {q.get('answer', '')}")
        lines.append(f"Rationale: {q.get('rationale', '')}")
        lines.append(f"Citations: {_format_citations(q.get('citations', []))}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def export_json(output: QuizOutput, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
