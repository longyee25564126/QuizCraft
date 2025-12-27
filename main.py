import argparse
import os

from quizcraft.config import Settings
from quizcraft.export import export_json, export_markdown, export_text
from quizcraft.pipeline import run_pipeline
from quizcraft.quiz import run_quiz
from quizcraft.utils import log_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuizCraft: Lecture PDF -> Summary -> Quiz")
    parser.add_argument("--pdf", required=True, help="Path to lecture PDF or .txt")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--quiz", action="store_true", help="Run interactive quiz after generation")

    parser.add_argument("--n-questions", type=int, default=None, help="Number of questions")
    parser.add_argument("--question-count", type=int, default=None, help="Number of questions (alias)")
    parser.add_argument("--question-types", default=None, help="Comma-separated types (tf,mcq)")

    parser.add_argument("--summary-len", type=int, default=None, help="Target summary length (chars)")
    parser.add_argument("--summary-min", type=int, default=None, help="Minimum summary length (chars)")
    parser.add_argument("--summary-max", type=int, default=None, help="Maximum summary length (chars)")

    parser.add_argument("--chunk-chars", type=int, default=None, help="Chunk target chars")
    parser.add_argument("--overlap-chars", type=int, default=None, help="Chunk overlap chars")

    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--top-k-chunks", type=int, default=None, help="Top-K chunks for selector")
    parser.add_argument("--long-doc-threshold-pages", type=int, default=None, help="Pages threshold to enable selector")

    parser.add_argument("--base-url", dest="base_url", default=None, help="Ollama base URL")
    parser.add_argument("--ollama-base-url", dest="base_url", default=None, help="Ollama base URL")
    parser.add_argument("--chat-model", default=None, help="Ollama chat model")
    parser.add_argument("--embed-model", default=None, help="Ollama embedding model")
    parser.add_argument("--max-context", type=int, default=None, help="Max context chars for summary")
    parser.add_argument("--max-input", type=int, default=None, help="Max chars per LLM input")
    parser.add_argument("--chat-timeout", type=int, default=None, help="Chat request timeout (seconds)")
    parser.add_argument("--reduce-timeout", type=int, default=None, help="Reduce summary timeout (seconds)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_types = None
    if args.question_types:
        question_types = [q.strip() for q in args.question_types.split(",") if q.strip()]

    question_count = args.n_questions if args.n_questions is not None else args.question_count

    summary_min = args.summary_min
    summary_max = args.summary_max
    if args.summary_len is not None:
        summary_min = max(50, args.summary_len - 10)
        summary_max = args.summary_len + 10

    settings = Settings.from_args(
        base_url=args.base_url,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        question_count=question_count,
        question_types=question_types,
        summary_min_chars=summary_min,
        summary_max_chars=summary_max,
        max_context_chars=args.max_context,
        max_input_chars=args.max_input,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        long_doc_threshold_pages=args.long_doc_threshold_pages,
        top_k_chunks=args.top_k_chunks,
        max_chunks=args.max_chunks,
        chat_timeout=args.chat_timeout,
        reduce_timeout=args.reduce_timeout,
        seed=args.seed,
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
