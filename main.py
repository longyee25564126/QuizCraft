import argparse
import os

from quizcraft.config import Settings
from quizcraft.export import export_json, export_markdown, export_text
from quizcraft.pipeline import run_pipeline
from quizcraft.quiz import run_quiz
from quizcraft.utils import log_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuizCraft: Lecture PDF -> Summary -> Quiz")
    parser.add_argument("--pdf", required=True, help="Path to lecture PDF")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--quiz", action="store_true", help="Run interactive quiz after generation")
    parser.add_argument("--question-count", type=int, default=None, help="Number of questions")
    parser.add_argument("--question-types", default=None, help="Comma-separated types (tf,mcq)")
    parser.add_argument("--summary-min", type=int, default=None, help="Minimum summary length (chars)")
    parser.add_argument("--summary-max", type=int, default=None, help="Maximum summary length (chars)")
    parser.add_argument("--base-url", default=None, help="Ollama base URL")
    parser.add_argument("--chat-model", default=None, help="Ollama chat model")
    parser.add_argument("--embed-model", default=None, help="Ollama embedding model")
    parser.add_argument("--max-context", type=int, default=None, help="Max context chars for summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_types = None
    if args.question_types:
        question_types = [q.strip() for q in args.question_types.split(",") if q.strip()]

    settings = Settings.from_args(
        base_url=args.base_url,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        question_count=args.question_count,
        question_types=question_types,
        summary_min_chars=args.summary_min,
        summary_max_chars=args.summary_max,
        max_context_chars=args.max_context,
    )

    output = run_pipeline(args.pdf, settings)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "output.json")
    md_path = os.path.join(args.out_dir, "output.md")
    txt_path = os.path.join(args.out_dir, "output.txt")

    log_step("Export outputs")
    export_json(output, json_path)
    export_markdown(output, md_path)
    export_text(output, txt_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {txt_path}")

    if args.quiz:
        run_quiz(output["questions"])


if __name__ == "__main__":
    main()
