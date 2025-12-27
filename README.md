# QuizCraft

Lecture PDF → Summary → Quiz. An AI tutor that generates evidence-backed questions and step-by-step explanations.

## Features
- PDF clean-up (low-info lines + repeated header/footer removal)
- Optional OCR fallback for scanned pages
- Structured chunking with section titles + page metadata
- Long-doc safe pipeline (selector + map-reduce)
- 章節摘要：overview + 3–6 sections（每段附 citations）+ 5–8 keypoints
- 5 題（可調）T/F、MCQ（可選 short/calc）題庫，附引用證據 + evidence quotes
- Verify + rewrite：證據不足就重寫/重生題目
- Quiz 模式：逐題作答、給提示、顯示解析與引用
- Export：output.json / output.md / output.txt

## Requirements
- Python 3.11+
- Ollama running locally (default: `http://localhost:11434`)
- Models: `llama3.1:8b-instruct-q8_0`, `nomic-embed-text:v1.5`

## Quickstart (Local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pull models
ollama pull llama3.1:8b-instruct-q8_0
ollama pull nomic-embed-text:v1.5

# run
python main.py --pdf /path/to/lecture.pdf --quiz
```

Outputs are saved to `outputs/output.json`, `outputs/output.md`, and `outputs/output.txt`.

## Summary Output Format
```json
{
  "summary": {
    "overview": "...",
    "sections": [
      {
        "title": "...",
        "summary": "...",
        "citations": [{"page": 3, "chunk_id": "p3_c2"}]
      }
    ],
    "keypoints": ["...", "..."]
  }
}
```

## Options
```bash
python main.py \
  --pdf /path/to/lecture.pdf \
  --n-questions 5 \
  --question-types tf,mcq \
  --summary-len 120 \
  --chunk-tokens 450 \
  --overlap-tokens 60 \
  --min-chunk-tokens 80 \
  --top-k-chunks 60 \
  --long-doc-threshold-pages 30 \
  --summary-budget-tokens 1200 \
  --evidence-budget-tokens 450 \
  --pages 1-10 \
  --max-pages 30 \
  --chapter "第3章" \
  --ollama-base-url http://localhost:11434 \
  --chat-timeout 120 \
  --reduce-timeout 240 \
  --quiz
```

Environment variables:
- `OLLAMA_BASE_URL`
- `QUIZCRAFT_CHAT_MODEL`
- `QUIZCRAFT_EMBED_MODEL`
- `QUIZCRAFT_CHUNK_TOKENS`, `QUIZCRAFT_OVERLAP_TOKENS`, `QUIZCRAFT_MIN_CHUNK_TOKENS`
- `QUIZCRAFT_TOP_K_CHUNKS`, `QUIZCRAFT_MAX_CHUNKS`, `QUIZCRAFT_LONG_DOC_PAGES`
- `QUIZCRAFT_SUMMARY_BUDGET_TOKENS`, `QUIZCRAFT_EVIDENCE_BUDGET_TOKENS`
- `QUIZCRAFT_EMBED_CACHE`, `QUIZCRAFT_EMBED_CACHE_ENABLED`
- `QUIZCRAFT_ENABLE_OCR`, `QUIZCRAFT_OCR_ENGINE`, `QUIZCRAFT_OCR_LANG`

## OCR (Optional)
To enable OCR fallback for scanned PDFs:
```bash
pip install pytesseract pdf2image
# system-level deps may be required (tesseract-ocr, poppler)
python main.py --pdf /path/to/scan.pdf --enable-ocr
```

## Podman + Ollama Container (Demo Setup)
### 1) Start Ollama container (if not running)
```bash
podman pod create --name ai-pod -p 11434:11434
podman run -d --name ollama --pod ai-pod -v ollama:/root/.ollama ollama/ollama
podman exec -it ollama ollama pull llama3.1:8b-instruct-q8_0
podman exec -it ollama ollama pull nomic-embed-text:v1.5
```

### 2) Run QuizCraft with Podman (CLI)
```bash
podman build -t quizcraft -f Containerfile .
podman run --rm -it \
  --add-host=host.containers.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.containers.internal:11434 \
  -v "$PWD:/app:Z" \
  quizcraft --pdf /app/lecture.pdf --quiz
```

### 3) Shortcut script
```bash
./run_podman.sh /app/lecture.pdf /app/outputs
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes
- If running inside a container, `OLLAMA_BASE_URL` defaults to `http://host.containers.internal:11434`.
- Summary sections include citations in the format `p{page}:{chunk_id}`.
- Long documents use selector + map-reduce; reduce step never re-feeds the full PDF.
