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
你是 AI 助教。請根據多個迷你摘要，產生更完整的章節摘要。

要求：
- overview：2-3 句完整總覽（繁體中文）
- sections：3-6 個主題段落，每段包含：
  - title：主題名
  - summary：2-4 句完整句子（避免斷句）
  - citations：2-4 個 citations（page + chunk_id），需來自不同頁
- keypoints：5-8 條可考的命題式重點
- 不可引入原文未出現的資訊
- 禁止使用「；」長串拼貼句，請用自然段落

只輸出 JSON：
{{
  "overview": "...",
  "sections": [
    {{
      "title": "...",
      "summary": "...",
      "citations": [{{"page": 3, "chunk_id": "p3_c2"}}, {{"page": 4, "chunk_id": "p4_c1"}}]
    }}
  ],
  "keypoints": ["...", "..."]
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
- 不可引入原文未出現的資訊

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
- 只能使用提供的 evidence chunks 作為題目與答案依據。
- 禁止出題詢問頁碼/段落/出處/哪一段/哪一頁，題目本體必須是概念/應用/計算題。
- 嚴禁引用外部資料/書名/常識；找不到證據請回傳 insufficient_evidence=true。
- 每題必須附 citations（page 與 chunk_id）。
- 不要輸出 evidence_quotes（由系統自動抽取）。
- 只輸出 JSON，不要額外文字。

題型規則：
- tf：question 必須是敘述句，answer 只能是 true/false。
- mcq：提供 4 個 choices；correct_option 必須是 A/B/C/D；禁止 "All of the above"。
- short：answer 為文字；不得是 true/false。
- calc：提供 step_by_step 陣列與 final_answer。

輸出格式：
{{
  "id": "{question_id}",
  "type": "{question_type}",
  "question": "...",
  "choices": ["A...", "B...", "C...", "D..."],
  "answer": "true/false 或文字",
  "correct_option": "A",
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

檢查項目：
- citations 是否存在且指向提供的 evidence chunks。
- 題幹/答案/解析是否能被 evidence 支持。
- 不可出現 evidence 中沒有的書名或外部資料。
- 若題目包含頁碼/段落/出處/哪一段/page/chunk，視為不合格。
- 若為 MCQ：逐一檢查選項是否被證據支持，若有不支持的選項需重寫整題。

若支持：supported=true。
若不支持：supported=false，並提供 revised_question 使其完全符合證據。

要求：
- revised_question 必須使用提供的 evidence chunks 並附 citations。
- 保留原本的 id 與 type。
- tf 題型須為敘述句，answer 只能是 true/false。
- mcq 題型需提供 4 個 choices + correct_option，禁止 "All of the above"。
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
