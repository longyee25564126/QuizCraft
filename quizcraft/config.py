import os
from dataclasses import dataclass
from typing import List


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _default_base_url() -> str:
    env_url = os.getenv("OLLAMA_BASE_URL")
    if env_url:
        return env_url
    if os.path.exists("/run/.containerenv") or os.path.exists("/.dockerenv"):
        return "http://host.containers.internal:11434"
    return "http://localhost:11434"


@dataclass
class Settings:
    base_url: str = _default_base_url()
    chat_model: str = _env("QUIZCRAFT_CHAT_MODEL", "llama3.1:8b-instruct-q8_0")
    embed_model: str = _env("QUIZCRAFT_EMBED_MODEL", "nomic-embed-text:v1.5")

    chunk_chars: int = _env_int("QUIZCRAFT_CHUNK_CHARS", 1000)
    overlap_chars: int = _env_int("QUIZCRAFT_OVERLAP_CHARS", 120)

    summary_min_chars: int = _env_int("QUIZCRAFT_SUMMARY_MIN", 100)
    summary_max_chars: int = _env_int("QUIZCRAFT_SUMMARY_MAX", 150)

    question_count: int = _env_int("QUIZCRAFT_QUESTION_COUNT", 5)
    question_types: List[str] = None

    max_context_chars: int = _env_int("QUIZCRAFT_MAX_CONTEXT_CHARS", 12000)
    max_input_chars: int = _env_int("QUIZCRAFT_MAX_INPUT_CHARS", 12000)

    evidence_per_keypoint: int = _env_int("QUIZCRAFT_EVIDENCE_PER_KP", 2)
    verify_retries: int = _env_int("QUIZCRAFT_VERIFY_RETRIES", 1)
    summary_retries: int = _env_int("QUIZCRAFT_SUMMARY_RETRIES", 1)
    question_retries: int = _env_int("QUIZCRAFT_QUESTION_RETRIES", 2)

    long_doc_threshold_pages: int = _env_int("QUIZCRAFT_LONG_DOC_PAGES", 30)
    selector_chunk_threshold: int = _env_int("QUIZCRAFT_SELECTOR_CHUNK_THRESHOLD", 80)
    top_k_chunks: int = _env_int("QUIZCRAFT_TOP_K_CHUNKS", 60)
    max_chunks: int = _env_int("QUIZCRAFT_MAX_CHUNKS", 120)

    chat_timeout: int = _env_int("QUIZCRAFT_CHAT_TIMEOUT", 90)
    reduce_timeout: int = _env_int("QUIZCRAFT_REDUCE_TIMEOUT", 180)

    seed: int = _env_int("QUIZCRAFT_SEED", 42)

    def __post_init__(self) -> None:
        if self.question_types is None:
            self.question_types = ["tf", "mcq"]

    @classmethod
    def from_args(
        cls,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        question_count: int | None = None,
        question_types: List[str] | None = None,
        summary_min_chars: int | None = None,
        summary_max_chars: int | None = None,
        max_context_chars: int | None = None,
        max_input_chars: int | None = None,
        chunk_chars: int | None = None,
        overlap_chars: int | None = None,
        long_doc_threshold_pages: int | None = None,
        top_k_chunks: int | None = None,
        max_chunks: int | None = None,
        chat_timeout: int | None = None,
        reduce_timeout: int | None = None,
        seed: int | None = None,
    ) -> "Settings":
        settings = cls()
        if base_url:
            settings.base_url = base_url
        if chat_model:
            settings.chat_model = chat_model
        if embed_model:
            settings.embed_model = embed_model
        if question_count is not None:
            settings.question_count = question_count
        if question_types:
            settings.question_types = question_types
        if summary_min_chars is not None:
            settings.summary_min_chars = summary_min_chars
        if summary_max_chars is not None:
            settings.summary_max_chars = summary_max_chars
        if max_context_chars is not None:
            settings.max_context_chars = max_context_chars
        if max_input_chars is not None:
            settings.max_input_chars = max_input_chars
        if chunk_chars is not None:
            settings.chunk_chars = chunk_chars
        if overlap_chars is not None:
            settings.overlap_chars = overlap_chars
        if long_doc_threshold_pages is not None:
            settings.long_doc_threshold_pages = long_doc_threshold_pages
        if top_k_chunks is not None:
            settings.top_k_chunks = top_k_chunks
        if max_chunks is not None:
            settings.max_chunks = max_chunks
        if chat_timeout is not None:
            settings.chat_timeout = chat_timeout
        if reduce_timeout is not None:
            settings.reduce_timeout = reduce_timeout
        if seed is not None:
            settings.seed = seed
        return settings
