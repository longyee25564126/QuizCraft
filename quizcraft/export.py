import json
from typing import Dict

from quizcraft.schemas import QuizOutput


def _format_citations(citations) -> str:
    if not citations:
        return "N/A"
    return ", ".join(f"p{c['page']}:{c['chunk_id']}" for c in citations)


def _format_quotes(quotes) -> str:
    if not quotes:
        return "N/A"
    return " / ".join(q.get("quote", "") for q in quotes if q.get("quote"))


def export_markdown(output: QuizOutput, path: str) -> None:
    lines = []
    lines.append("# QuizCraft Output\n")
    lines.append("## Summary\n")
    summary = output.get("summary", {})

    if isinstance(summary, dict):
        overview = summary.get("overview", "")
        sections = summary.get("sections", [])
        keypoints = summary.get("keypoints", [])

        lines.append("### Overview\n")
        lines.append(overview + "\n")

        lines.append("### Sections\n")
        for idx, section in enumerate(sections, start=1):
            lines.append(f"#### {idx}. {section.get('title', '')}")
            lines.append(section.get("summary", ""))
            lines.append(f"**Citations:** {_format_citations(section.get('citations', []))}")
            lines.append("")

        lines.append("### Keypoints\n")
        for kp in keypoints:
            lines.append(f"- {kp}")
        lines.append("")
    else:
        lines.append(str(summary) + "\n")

    lines.append("## Questions\n")
    for idx, q in enumerate(output["questions"], start=1):
        lines.append(f"### Q{idx}: {q.get('question', '')}")
        if q.get("type") == "mcq":
            for i, choice in enumerate(q.get("choices", []), start=1):
                letter = chr(ord("A") + i - 1)
                lines.append(f"- {letter}. {choice}")
            if q.get("correct_option"):
                lines.append(f"**Correct Option:** {q.get('correct_option')} ")
        lines.append(f"**Answer:** {q.get('answer', '')}")
        if q.get("type") == "calc":
            lines.append(f"**Step-by-step:** {' / '.join(q.get('step_by_step', []))}")
            lines.append(f"**Final Answer:** {q.get('final_answer', '')}")
        lines.append(f"**Rationale:** {q.get('rationale', '')}")
        lines.append(f"**Citations:** {_format_citations(q.get('citations', []))}")
        lines.append(f"**Evidence Quotes:** {_format_quotes(q.get('evidence_quotes', []))}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def export_text(output: QuizOutput, path: str) -> None:
    lines = []
    lines.append("QuizCraft Output\n")
    lines.append("Summary:\n")
    summary = output.get("summary", {})

    if isinstance(summary, dict):
        overview = summary.get("overview", "")
        sections = summary.get("sections", [])
        keypoints = summary.get("keypoints", [])

        lines.append("Overview:\n")
        lines.append(overview + "\n")

        lines.append("Sections:\n")
        for idx, section in enumerate(sections, start=1):
            lines.append(f"[{idx}] {section.get('title', '')}")
            lines.append(section.get("summary", ""))
            lines.append(f"Citations: {_format_citations(section.get('citations', []))}")
            lines.append("")

        lines.append("Keypoints:\n")
        for kp in keypoints:
            lines.append(f"- {kp}")
        lines.append("")
    else:
        lines.append(str(summary) + "\n")

    lines.append("Questions:\n")
    for idx, q in enumerate(output["questions"], start=1):
        lines.append(f"Q{idx}: {q.get('question', '')}")
        if q.get("type") == "mcq":
            for i, choice in enumerate(q.get("choices", []), start=1):
                letter = chr(ord("A") + i - 1)
                lines.append(f"  {letter}. {choice}")
            if q.get("correct_option"):
                lines.append(f"Correct Option: {q.get('correct_option')}")
        lines.append(f"Answer: {q.get('answer', '')}")
        if q.get("type") == "calc":
            lines.append(f"Step-by-step: {' / '.join(q.get('step_by_step', []))}")
            lines.append(f"Final Answer: {q.get('final_answer', '')}")
        lines.append(f"Rationale: {q.get('rationale', '')}")
        lines.append(f"Citations: {_format_citations(q.get('citations', []))}")
        lines.append(f"Evidence Quotes: {_format_quotes(q.get('evidence_quotes', []))}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def export_json(output: QuizOutput, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
