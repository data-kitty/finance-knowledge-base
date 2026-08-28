"""
导入流程提示词模板。

包含以下 LLM 提示词:
- NAME_RECOGNITION: 从文件标题和切片内容中识别金融主题名称
  (例如 "个人贷款", "理财", "信用卡")。
- SYSTEM_PROMPT: 主题名称识别 LLM 调用的系统消息。
"""

SYSTEM_PROMPT = (
    "You are a professional financial topic recognition model. "
    "Based on the provided information, identify the finance topic name."
)

NAME_RECOGNITION = """
Please identify the finance topic name from the following information:

File name: {file_title}

Text chunks (for context):
{context}

Requirements:
1. Return the result as a string, ideally a complete finance topic name.
   Examples: "个人贷款", "理财", "信用卡", "账户与交易", "风控与反洗钱", "金融常见问题".
2. The result should only contain the finance topic name, no explanations or other content.
3. If the finance topic name cannot be identified, return an empty string.
"""
