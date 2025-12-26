SUMMARY_PROMPT = """
你是 AI 助教。請根據以下講義內容，產生繁體中文的摘要與重點。

需求：
- summary：{summary_min} 到 {summary_max} 字（繁體中文）
- keypoints：3 到 5 點（繁體中文）

只輸出 JSON，格式如下：
{{
  "summary": "...",
  "keypoints": ["...", "..."]
}}

講義內容：
{context}
"""

QUESTION_PROMPT = """
你是 AI 助教，根據以下重點與證據段落產生題庫。

規則：
- 只可使用提供的證據段落作為題目與答案依據。
- 每題必須附 citations（page 與 chunk_id），引用你使用的證據段落。
- rationale 必須包含原文片段（可直接引用原文），並用 2-3 句繁體中文解釋。
- 題型僅限：{question_types}。
- mcq 題必須提供 4 個 choices。
- 只輸出 JSON，不要額外文字。

輸出格式：
{{
  "questions": [
    {{
      "id": "q1",
      "type": "tf" | "mcq",
      "question": "...",
      "choices": ["A...", "B...", "C...", "D..."],
      "answer": "true/false" 或 "B",
      "rationale": "...",
      "citations": [{{"page": 3, "chunk_id": "p3_c2"}}],
      "difficulty": "easy|medium|hard",
      "concept_tags": ["..."]
    }}
  ]
}}

重點：
{keypoints}

證據段落（只能從這些段落取證）：
{evidence}

請產生 {question_count} 題。
"""

VERIFY_PROMPT = """
你是嚴格的事實查核員。請檢查題目是否能被證據段落支持。

若支持：supported=true。
若不支持：supported=false，並提供 revised_question 使其完全符合證據。

要求：
- revised_question 必須使用提供的證據段落並附 citations。
- 保留原本的 id 與 type。
- rationale 必須包含原文片段與 2-3 句繁體中文解釋。
- 只輸出 JSON。

輸出格式：
{{
  "supported": true/false,
  "reason": "...",
  "revised_question": null 或 {{...question object...}}
}}

題目 JSON：
{question_json}

證據段落：
{evidence}
"""
