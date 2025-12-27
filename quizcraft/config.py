import os
from dataclasses import dataclass
from typing import List, Optional, Set


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


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y"}


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

    chunk_tokens: int = _env_int("QUIZCRAFT_CHUNK_TOKENS", 450)
    overlap_tokens: int = _env_int("QUIZCRAFT_OVERLAP_TOKENS", 60)
    min_chunk_tokens: int = _env_int("QUIZCRAFT_MIN_CHUNK_TOKENS", 80)
    chunk_chars: int = _env_int("QUIZCRAFT_CHUNK_CHARS", 1000)
    overlap_chars: int = _env_int("QUIZCRAFT_OVERLAP_CHARS", 120)

    summary_min_chars: int = _env_int("QUIZCRAFT_SUMMARY_MIN", 100)
    summary_max_chars: int = _env_int("QUIZCRAFT_SUMMARY_MAX", 150)

    question_count: int = _env_int("QUIZCRAFT_QUESTION_COUNT", 5)
    question_types: List[str] = None

    max_context_chars: int = _env_int("QUIZCRAFT_MAX_CONTEXT_CHARS", 12000)
    max_input_chars: int = _env_int("QUIZCRAFT_MAX_INPUT_CHARS", 12000)
    summary_budget_tokens: int = _env_int("QUIZCRAFT_SUMMARY_BUDGET_TOKENS", 1200)
    evidence_budget_tokens: int = _env_int("QUIZCRAFT_EVIDENCE_BUDGET_TOKENS", 450)

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

    header_footer_threshold: int = _env_int("QUIZCRAFT_HEADER_FOOTER_THRESHOLD", 0)
    enable_ocr: bool = _env_bool("QUIZCRAFT_ENABLE_OCR", False)
    ocr_engine: str = _env("QUIZCRAFT_OCR_ENGINE", "tesseract")
    ocr_lang: str = _env("QUIZCRAFT_OCR_LANG", "chi_tra+eng")

    pages_filter: Optional[Set[int]] = None
    max_pages: Optional[int] = None
    chapter_filter: Optional[str] = None

    embed_cache_dir: str = _env("QUIZCRAFT_EMBED_CACHE", "outputs/.cache")
    embed_cache_enabled: bool = _env_bool("QUIZCRAFT_EMBED_CACHE_ENABLED", True)

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
        chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_chunk_tokens: int | None = None,
        long_doc_threshold_pages: int | None = None,
        top_k_chunks: int | None = None,
        max_chunks: int | None = None,
        chat_timeout: int | None = None,
        reduce_timeout: int | None = None,
        summary_budget_tokens: int | None = None,
        evidence_budget_tokens: int | None = None,
        enable_ocr: bool | None = None,
        ocr_engine: str | None = None,
        ocr_lang: str | None = None,
        pages_filter: Optional[Set[int]] = None,
        max_pages: Optional[int] = None,
        chapter_filter: Optional[str] = None,
        embed_cache_enabled: bool | None = None,
        embed_cache_dir: str | None = None,
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
        if chunk_tokens is not None:
            settings.chunk_tokens = chunk_tokens
        if overlap_tokens is not None:
            settings.overlap_tokens = overlap_tokens
        if min_chunk_tokens is not None:
            settings.min_chunk_tokens = min_chunk_tokens
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
        if summary_budget_tokens is not None:
            settings.summary_budget_tokens = summary_budget_tokens
        if evidence_budget_tokens is not None:
            settings.evidence_budget_tokens = evidence_budget_tokens
        if enable_ocr is not None:
            settings.enable_ocr = enable_ocr
        if ocr_engine:
            settings.ocr_engine = ocr_engine
        if ocr_lang:
            settings.ocr_lang = ocr_lang
        if pages_filter is not None:
            settings.pages_filter = pages_filter
        if max_pages is not None:
            settings.max_pages = max_pages
        if chapter_filter:
            settings.chapter_filter = chapter_filter
        if embed_cache_enabled is not None:
            settings.embed_cache_enabled = embed_cache_enabled
        if embed_cache_dir:
            settings.embed_cache_dir = embed_cache_dir
        if seed is not None:
            settings.seed = seed

        if chunk_tokens is None and chunk_chars is not None:
            settings.chunk_tokens = max(200, int(settings.chunk_chars * 0.45))
        if overlap_tokens is None and overlap_chars is not None:
            settings.overlap_tokens = max(20, int(settings.overlap_chars * 0.45))

        return settings
