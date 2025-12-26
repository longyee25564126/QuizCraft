# QuizCraft

Lecture PDF → Summary → Quiz. An AI tutor that generates evidence-backed questions and step-by-step explanations.

## Features
- PDF ingest with page metadata and chunk IDs
- 100–150 字繁中摘要 + 3–5 重點條列
- 5 題（可調）T/F 或 MCQ 題庫，附引用證據與解析
- Verify + rewrite：證據不足就重寫題目
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

## Options
```bash
python main.py \
  --pdf /path/to/lecture.pdf \
  --question-count 5 \
  --question-types tf,mcq \
  --summary-min 100 \
  --summary-max 150 \
  --quiz
```

Environment variables:
- `OLLAMA_BASE_URL`
- `QUIZCRAFT_CHAT_MODEL`
- `QUIZCRAFT_EMBED_MODEL`

## Container (Optional)
```bash
podman build -t quizcraft -f Containerfile .
podman run --rm -it \
  --add-host=host.containers.internal:host-gateway \
  -v "$PWD:/app:Z" \
  quizcraft --pdf /app/lecture.pdf --quiz
```
