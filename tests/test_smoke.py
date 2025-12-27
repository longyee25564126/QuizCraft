import re
from pathlib import Path

import quizcraft.pipeline as pipeline
from quizcraft.config import Settings


class FakeOllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def check_health(self) -> bool:
        return True

    def embed(self, model: str, text: str):
        return [float(len(text) or 1)]

    def chat_json(self, model: str, messages, options=None):
        prompt = messages[0]["content"]
        if "[MAP_SUMMARY]" in prompt:
            return {
                "mini_summary": "這是單段摘要。",
                "keywords": ["代理人", "目標"],
                "citations": [{"page": 1, "chunk_id": "p1_c1"}],
            }
        if "[REDUCE_SUMMARY]" in prompt:
            return {
                "summary": "這份講義介紹代理人如何感知環境並依目標行動與調整策略。" * 6,
                "keypoints": [
                    "代理人能感知環境並採取行動",
                    "代理人會根據目標決策",
                    "代理人可依回饋調整策略",
                ],
                "citations": [{"page": 1, "chunk_id": "p1_c1"}],
            }
        if "[CONCEPT_EXTRACT]" in prompt:
            return {
                "concepts": [
                    {
                        "name": "代理人感知",
                        "description": "代理人能感知環境",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                        "difficulty": "easy",
                    },
                    {
                        "name": "目標導向",
                        "description": "依目標決策",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                        "difficulty": "easy",
                    },
                    {
                        "name": "回饋調整",
                        "description": "能依回饋調整策略",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                        "difficulty": "medium",
                    },
                    {
                        "name": "代理人循環",
                        "description": "感知、推理、行動",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                        "difficulty": "medium",
                    },
                    {
                        "name": "行動與學習",
                        "description": "行動與學習形成循環",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                        "difficulty": "medium",
                    },
                ]
            }
        if "[QUESTION_GENERATION]" in prompt:
            return {
                "id": "q1",
                "type": "tf",
                "question": "代理人能感知環境並採取行動。",
                "answer": "true",
                "rationale": "原文說明代理人能感知環境並做出行動。",
                "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                "difficulty": "easy",
                "concept_tags": ["代理人"],
            }
        if "[VERIFY_QUESTION]" in prompt:
            return {"supported": True, "reason": "ok", "revised_question": None}
        return {}


def test_pipeline_smoke(monkeypatch):
    monkeypatch.setattr(pipeline, "OllamaClient", FakeOllamaClient)
    sample_path = Path(__file__).parent / "data" / "sample.txt"

    settings = Settings.from_args(base_url="http://fake", question_count=5)
    output = pipeline.run_pipeline(str(sample_path), settings)

    assert settings.summary_min_chars <= len(output["summary"]) <= settings.summary_max_chars + 20
    assert len(output["keypoints"]) >= 3
    assert len(output["questions"]) >= settings.question_count

    assert re.search(r"p\d+_c\d+", output["summary"])
    for keypoint in output["keypoints"]:
        assert re.search(r"p\d+_c\d+", keypoint)

    for question in output["questions"]:
        assert question.get("id")
        assert question.get("type")
        assert question.get("question")
        assert question.get("answer")
        assert question.get("rationale")
        citations = question.get("citations", [])
        assert citations and citations[0].get("page") is not None
