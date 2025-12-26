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


@dataclass
class Settings:
    base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = _env("QUIZCRAFT_CHAT_MODEL", "llama3.1:8b-instruct-q8_0")
    embed_model: str = _env("QUIZCRAFT_EMBED_MODEL", "nomic-embed-text:v1.5")
    chunk_size: int = _env_int("QUIZCRAFT_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("QUIZCRAFT_CHUNK_OVERLAP", 120)
    summary_min_chars: int = _env_int("QUIZCRAFT_SUMMARY_MIN", 100)
    summary_max_chars: int = _env_int("QUIZCRAFT_SUMMARY_MAX", 150)
    question_count: int = _env_int("QUIZCRAFT_QUESTION_COUNT", 5)
    question_types: List[str] = None
    max_context_chars: int = _env_int("QUIZCRAFT_MAX_CONTEXT_CHARS", 12000)
    evidence_per_keypoint: int = _env_int("QUIZCRAFT_EVIDENCE_PER_KP", 2)
    verify_retries: int = _env_int("QUIZCRAFT_VERIFY_RETRIES", 1)

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
        return settings
