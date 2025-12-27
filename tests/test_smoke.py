import re
from pathlib import Path

import quizcraft.pipeline as pipeline
from quizcraft.config import Settings


class FakeOllamaClient:
    def __init__(self, base_url: str, default_timeout: int = 60) -> None:
        self.base_url = base_url
        self.default_timeout = default_timeout

    def check_health(self) -> bool:
        return True

    def embed(self, model: str, text: str):
        return [float(len(text) or 1)]

    def chat_json(self, model: str, messages, options=None, timeout=None):
        prompt = messages[0]["content"]
        if "[MAP_SUMMARY]" in prompt:
            return {
                "mini_summary": "這是單段摘要，描述代理人如何感知環境並採取行動。",
                "keywords": ["代理人", "目標"],
                "citations": [{"page": 1, "chunk_id": "p1_c1"}],
            }
        if "[REDUCE_SUMMARY]" in prompt:
            return {
                "overview": "本章說明代理人的核心循環與目標導向決策，並指出回饋調整的重要性。代理人透過感知、推理與行動形成學習迴圈。",
                "sections": [
                    {
                        "title": "代理人核心循環",
                        "summary": "本段說明代理人感知環境並採取行動。強調感知、推理與行動的循環。",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                    },
                    {
                        "title": "目標與決策",
                        "summary": "本段描述代理人依目標做出決策。並可透過回饋調整策略。",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                    },
                    {
                        "title": "學習與調整",
                        "summary": "本段強調回饋後的策略調整機制。形成持續學習的循環。",
                        "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                    },
                ],
                "keypoints": [
                    "代理人能感知環境並採取行動",
                    "代理人依目標做出決策",
                    "回饋可促進策略調整",
                    "感知、推理與行動形成循環",
                    "學習過程強調持續改進",
                ],
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
            if '"type": "mcq"' in prompt:
                return {
                    "id": "q1",
                    "type": "mcq",
                    "question": "下列何者正確描述代理人？",
                    "choices": ["能感知環境", "只能被動回應", "不會調整策略", "與環境無互動"],
                    "answer": "A",
                    "correct_option": "A",
                    "rationale": "原文提到代理人能感知環境並採取行動。",
                    "citations": [{"page": 1, "chunk_id": "p1_c1"}],
                    "difficulty": "easy",
                    "concept_tags": ["代理人"],
                }
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

    settings = Settings.from_args(base_url="http://fake", question_count=5, question_types=["tf", "mcq"])
    output = pipeline.run_pipeline(str(sample_path), settings)

    summary = output["summary"]
    assert summary.get("overview")
    assert len(summary.get("sections", [])) >= 3
    assert len(summary.get("keypoints", [])) >= 5

    for section in summary.get("sections", []):
        assert section.get("title")
        assert section.get("summary")
        citations = section.get("citations", [])
        assert citations

    assert len(output["questions"]) >= settings.question_count

    for question in output["questions"]:
        assert question.get("id")
        assert question.get("type")
        assert question.get("question")
        assert question.get("answer")
        assert question.get("rationale")
        citations = question.get("citations", [])
        assert citations and citations[0].get("page") is not None
        quotes = question.get("evidence_quotes", [])
        assert quotes
        for quote in quotes:
            assert 20 <= len(quote.get("quote", "")) <= 80
        if question.get("type") == "mcq":
            assert len(question.get("choices", [])) == 4
            assert question.get("correct_option") in {"A", "B", "C", "D"}
            assert question.get("answer") == question.get("correct_option")
        if question.get("type") == "tf":
            assert question.get("answer") in {"true", "false"}
            assert not question.get("question", "").strip().endswith("?")
