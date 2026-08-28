"""
DashScope 重排工具模块。

调用 DashScope TextReRank API 按与查询的相关性对文档重排。
所有配置 (api_key, model, instruct) 均从 rag.config 读取。
"""
import logging
from http import HTTPStatus
from typing import List

import dashscope

from finance_knowledge_rag.config import rag_config

logger = logging.getLogger(__name__)


def rerank_documents(query: str, documents: List[str]) -> List[float]:
    """
    调用 DashScope TextReRank API 按与查询的相关性对文档打分。

    Args:
        query: 用户查询字符串。
        documents: 待重排的文档文本列表。

    Returns:
        相关性分数列表 (float), 与输入文档顺序对齐。

    Raises:
        RuntimeError: 当重排 API 调用失败时。
    """
    if not documents:
        return []

    # 设置 DashScope API key
    dashscope.api_key = rag_config.reranker_http.api_key

    resp = dashscope.TextReRank.call(
        model=rag_config.reranker_http.model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=rag_config.reranker_http.instruct,
    )

    if resp.status_code != HTTPStatus.OK:
        message = resp.message
        raise RuntimeError(
            f"Reranker API call failed: status_code={resp.status_code}, message={message}"
        )

    results = resp.output.results

    # 构建与原始文档索引对齐的分数数组
    scores = [0.0] * len(documents)
    for item in results:
        index = item.index
        score = item.relevance_score
        scores[index] = score

    logger.info("Rerank completed for %d documents", len(documents))
    return scores
