# QuizCraft Containerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OLLAMA_BASE_URL=http://host.containers.internal:11434

ENTRYPOINT ["python", "main.py"]
