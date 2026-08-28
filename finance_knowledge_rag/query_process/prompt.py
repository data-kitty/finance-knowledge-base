"""
查询流程提示词模板。

包含以下 LLM 提示词:
- ITEM_NAME_EXTRACT_TEMPLATE: 从用户查询中提取金融主题名称
  (例如 "贷款", "理财", "信用卡") 并改写查询。
- ITEM_NAME_EXTRACT_SYSTEM_PROMPT: 主题提取的系统消息。
- ANSWER_PROMPT: 基于检索上下文回答用户的金融问题。
- HYDE_PROMPT: 为 HyDE 检索生成假设性答案。
"""

ITEM_NAME_EXTRACT_SYSTEM_PROMPT = (
    "You are a professional financial customer service assistant, skilled at "
    "understanding user intent and extracting key finance topic information. "
    "You can identify finance topics such as personal loans, wealth management, "
    "credit cards, accounts, risk control, etc."
)

ITEM_NAME_EXTRACT_TEMPLATE = """
Chat history:
{history_text}

Current user question:
{original_query}

Based on the chat history and current question, extract the finance topic(s) the user is asking about.

Important definitions:
- Finance topic: refers to a financial business category, such as "个人贷款" (personal loan),
  "理财" (wealth management), "信用卡" (credit card), "账户与交易" (account & transaction),
  "风控与反洗钱" (risk control & anti-money laundering), "金融常见问题" (FAQ).
- The topic name should be a broad category, not a specific product code.

Extraction rules:
1. If the user uses pronouns (e.g. "这个", "它"), resolve the reference based on chat history.
2. If the chat history contains topic information, include it in item_names.
3. If no finance topic can be identified from the question or history, return an empty list.
4. Rewrite the user's question (rewritten_query) to be a self-contained, complete question
   that includes the identified finance topic name(s).
5. There may be one or more topics, but no duplicates.

Return the result directly in JSON format:
{{
    "item_names": ["个人贷款", "理财"],
    "rewritten_query": "关于个人贷款和理财的相关问题"
}}
"""

HYDE_PROMPT = """
Based on the following user query, generate a concise answer draft.
User query: {rewritten_query}
Requirements:
1. The answer should be concise and clear, containing core information only.
2. Assume you are a finance domain expert, provide professional explanation.
3. Do not use uncertain words like "maybe" or "possibly".
4. Keep the answer highly relevant to the query topic.
5. Answer in Chinese, no more than 300 characters.
"""

ANSWER_PROMPT = """You are an intelligent financial customer service assistant. Based on the reference content, answer the user's question.

Requirements:
1. Try to answer based on the [Reference Content] and [User Question]. Do not fabricate facts.
2. If the user's question needs illustration (e.g. process, structure), describe it clearly in text.
3. Keep the answer professional, accurate, and helpful.

[Reference Content]
{context}

[Chat History]
{history}

[Related Finance Topic]
{item_names}

[User Question]
{question}

Please answer:"""
