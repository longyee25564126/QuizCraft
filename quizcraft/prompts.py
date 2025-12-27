MAP_SUMMARY_PROMPT = """
[MAP_SUMMARY]
你是 AI 助教。請針對單一講義段落輸出迷你摘要。

要求：
- mini_summary：1-2 句繁體中文
- keywords：2-5 個關鍵詞（繁體中文）
- citations：必須包含此段落的 page 與 chunk_id

只輸出 JSON：
{{
  "mini_summary": "...",
  "keywords": ["...", "..."],
  "citations": [{{"page": 3, "chunk_id": "p3_c2"}}]
}}

段落資訊：
page={page}
chunk_id={chunk_id}

段落內容：
{chunk_text}
"""

REDUCE_SUMMARY_PROMPT = """
[REDUCE_SUMMARY]
你是 AI 助教。請根據多個迷你摘要，產生整體摘要與重點。

要求：
- summary：{summary_min} 到 {summary_max} 字（繁體中文）
- keypoints：3-5 點（繁體中文）
- citations：合併/挑選與 summary/keypoints 相關的 citations
- 不可引入原文未出現的資訊

只輸出 JSON：
{{
  "summary": "...",
  "keypoints": ["...", "..."],
  "citations": [{{"page": 3, "chunk_id": "p3_c2"}}]
}}

迷你摘要清單（含 citations）：
{mini_summaries}
"""

CONCEPT_PROMPT = """
[CONCEPT_EXTRACT]
你是 AI 助教，請根據 keypoints 與迷你摘要抽出觀念清單。

要求：
- 每個 concept 需附 citations（page + chunk_id）
- 回傳 concepts 陣列，最多 {max_concepts} 個

只輸出 JSON：
{{
  "concepts": [
    {{
      "name": "...",
      "description": "...",
      "citations": [{{"page": 3, "chunk_id": "p3_c2"}}],
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Keypoints:
{keypoints}

Mini summaries:
{mini_summaries}
"""

QUESTION_PROMPT = """
[QUESTION_GENERATION]
你是 AI 助教，請根據單一 concept 與其證據段落出題。

規則：
- 只可使用提供的證據段落作為題目與答案依據。
- 每題必須附 citations（page 與 chunk_id）。
- rationale 必須包含原文片段，並用 2-3 句繁體中文解釋。
- 題型僅限：{question_types}。
- mcq 必須提供 4 個 choices。
- 只輸出 JSON，不要額外文字。

輸出格式：
{{
  "id": "{question_id}",
  "type": "tf" | "mcq",
  "question": "...",
  "choices": ["A...", "B...", "C...", "D..."],
  "answer": "true/false" 或 "B",
  "rationale": "...",
  "citations": [{{"page": 3, "chunk_id": "p3_c2"}}],
  "difficulty": "easy|medium|hard",
  "concept_tags": ["..."]
}}

Concept:
{concept}

Evidence chunks (only use these):
{evidence}
"""

VERIFY_PROMPT = """
[VERIFY_QUESTION]
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
  "revised_question": null 或 {{ ...question object... }}
}}

題目 JSON：
{question_json}

證據段落：
{evidence}
"""
