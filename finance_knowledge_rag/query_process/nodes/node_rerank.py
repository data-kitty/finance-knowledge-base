"""
查询工作流的重排节点。

调用 DashScope 重排器对 RRF 结果重新打分, 并通过悬崖检测
对 Top-K 做动态截断。
"""
import logging
from typing import Any, Dict, List

from finance_knowledge_rag.query_process.base import NodeBase
from finance_knowledge_rag.query_process.state import QueryGraphState
from finance_knowledge_rag.utils.reranker import rerank_documents

logger = logging.getLogger(__name__)


class NodeRerank(NodeBase):
    """
    重排节点: 使用 DashScope 重排器对文档精确重新打分。

    重排后应用带悬崖检测的动态 Top-K 截断,
    以移除低质量结果。
    """

    name = "node_rerank"

    # 动态 TopK 限制
    RERANK_MAX_TOPK = 10  # 硬性上限
    RERANK_MIN_TOPK = 2   # 硬性下限

    # 悬崖检测阈值
    RERANK_GAP_RATIO = 0.25  # 相对差距
    RERANK_GAP_ABS = 0.10    # 绝对差距

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        执行重排流水线。

        Args:
            state: 必须包含 'rrf_chunks' 和 'rewritten_query'。

        Returns:
            包含 'reranked_docs' 的字典 — Top-K 重排后的文档。
        """
        # 步骤 1: 合并多源文档 (本项目仅本地 RRF)
        merged_docs = self._step1_merge_docs(state)

        # 步骤 2: 重排
        reranked_docs = self._step2_rerank_docs(state, merged_docs)

        # 步骤 3: 带悬崖检测的动态 TopK 截断
        cutoff_docs = self._step3_cliff_cutoff(reranked_docs)

        logger.info("Rerank: %d input -> %d output", len(merged_docs), len(cutoff_docs))
        return {"reranked_docs": cutoff_docs}

    def _step1_merge_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """将 RRF 切片格式化为统一的文档结构。"""
        merged: List[Dict[str, Any]] = []

        for rrf_doc in state.get("rrf_chunks", []):
            formatted = {
                "title": rrf_doc.get("title"),
                "content": rrf_doc.get("content"),
                "chunk_id": rrf_doc.get("chunk_id"),
                "item_name": rrf_doc.get("item_name"),
                "url": None,
                "source": "local",
            }
            merged.append(formatted)

        return merged

    def _step2_rerank_docs(
        self, state: QueryGraphState, merged_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """调用 DashScope 重排器对文档打分并排序。"""
        user_query = state.get("rewritten_query", "")

        if not merged_docs:
            return []

        contents = [doc.get("content", "") for doc in merged_docs]

        # 调用重排器; 失败时降级为 RRF 原始顺序(score=0, 悬崖检测不触发截断),
        # 避免重排器不可用阻塞整个查询链路
        try:
            rerank_scores = rerank_documents(user_query, contents)
        except Exception as e:
            logger.warning(
                "Reranker unavailable (%s), fallback to RRF order", e
            )
            return [{"score": 0.0, **doc} for doc in merged_docs]

        # 附加分数
        reranked = [
            {"score": score, **doc}
            for doc, score in zip(merged_docs, rerank_scores)
        ]

        # 按分数降序排序
        sorted_docs = sorted(reranked, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_docs

    def _step3_cliff_cutoff(
        self, ranked_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        带悬崖检测的动态 TopK 截断。

        若相邻文档之间的分数差距超过阈值 (绝对或相对),
        则在断点处截断。
        """
        if not ranked_docs:
            return []

        upper_bound = min(self.RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(self.RERANK_MIN_TOPK, upper_bound)

        # 默认: 取至多 upper_bound 个
        cutoff_pos = upper_bound

        # 从 lower_bound 开始检测悬崖
        for index in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[index].get("score", 0)
            next_score = ranked_docs[index + 1].get("score", 0)

            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= self.RERANK_GAP_ABS or rel_gap >= self.RERANK_GAP_RATIO:
                cutoff_pos = index + 1
                logger.info("Cliff detected at position %d", cutoff_pos)
                break

        return ranked_docs[:cutoff_pos]
