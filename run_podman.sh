#!/bin/bash
set -euo pipefail

PDF_PATH="${1:-/app/lecture.pdf}"
OUT_DIR="${2:-/app/outputs}"

CHAT_TIMEOUT="${QUIZCRAFT_CHAT_TIMEOUT:-90}"
REDUCE_TIMEOUT="${QUIZCRAFT_REDUCE_TIMEOUT:-180}"


echo "Building QuizCraft image..."
podman build -t quizcraft -f Containerfile .

echo "Running QuizCraft CLI..."
podman run --rm -it \
  --add-host=host.containers.internal:host-gateway \
  -e OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.containers.internal:11434}" \
  -e QUIZCRAFT_CHAT_TIMEOUT="${CHAT_TIMEOUT}" \
  -e QUIZCRAFT_REDUCE_TIMEOUT="${REDUCE_TIMEOUT}" \
  -v "$(pwd):/app:Z" \
  -w /app \
  quizcraft --pdf "$PDF_PATH" --out-dir "$OUT_DIR" --quiz
