import json
from typing import Any, Dict, List, Optional

import requests

from quizcraft.utils import extract_json


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        format_json: bool = False,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if format_json:
            payload["format"] = "json"
        if options:
            payload["options"] = options
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def chat_json(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content = self.chat(model, messages, format_json=True, options=options)
        return extract_json(content)

    def embed(self, model: str, text: str, timeout: int = 60) -> List[float]:
        payload = {"model": model, "prompt": text}
        resp = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]
