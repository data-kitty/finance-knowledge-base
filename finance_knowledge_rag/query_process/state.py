"""
查询流程图状态定义。

定义 QueryGraphState TypedDict — 在查询 (检索 + 答案生成) 工作流
所有节点间流转的数据结构。
"""
from typing import TypedDict, List, Any


class QueryGraphState(TypedDict):
    """
    查询工作流的图状态。

    包含查询节点产生和消费的所有数据字段:
    会话跟踪、用户查询、检索结果、排序结果与答案。
    """
    # 会话/消息跟踪
    session_id: str  # 用于聊天历史的会话 ID
    message_id: str  # 消息 ID (MongoDB ObjectId 字符串)
    task_id: str  # 用于进度跟踪的 SSE 任务 ID

    # 用户查询
    original_query: str  # 用户的原始问题
    rewritten_query: str  # 改写后的问题 (已嵌入 item_names)

    # 金融主题
    item_names: List[str]  # 提取的金融主题名称 (例如 ["个人贷款", "理财"])

    # 检索结果
    embedding_chunks: List[Any]  # 向量检索结果

    # 排序结果
    rrf_chunks: List[Any]  # RRF 融合排序后的结果
    reranked_docs: List[Any]  # 重排 Top-K 后的结果

    # 生成
    prompt: str  # 组装好的 LLM 提示词
    answer: str  # 最终生成的答案

    # 历史
    history: List[Any]  # 来自 MongoDB 的聊天历史记录

    # 流式
    is_stream: bool  # 是否流式输出
