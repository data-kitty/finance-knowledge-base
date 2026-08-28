"""
查询工作流的 RRF (倒数排名融合) 节点。

本项目只有单条检索路径 (向量检索), 因此 RRF 主要执行
去重和排序。算法保持通用, 以支持未来新增路径的多路融合。
"""
import logging
from typing import Any, Dict, List, Tuple

from finance_knowledge_rag.query_process.base import NodeBase
from finance_knowledge_rag.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)


class NodeRrf(NodeBase):
    """
    倒数排名融合节点。

    融合并排序多路检索结果。当前仅使用向量检索路径,
    因此 RRF 承担去重 + 排序的职责。
    """

    name = "node_rrf"

    # 融合后返回的最大结果数
    MAX_RESULTS = 5
    # RRF 平滑常数
    K = 60

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        对 embedding_chunks 执行 RRF 融合。

        Args:
            state: 必须包含 'embedding_chunks'。

        Returns:
            包含 'rrf_chunks' 的字典 — 去重、排序后的切片列表。
        """
        # 1. 获取向量检索结果
        embedding_chunks = state.get("embedding_chunks", [])
        embedding_search_list = [
            doc.get("entity") for doc in embedding_chunks if isinstance(doc, dict)
        ]

        # 2. 定义带权重的 RRF 输入
        # 本项目仅单路径 (向量) — 权重 1.0
        rrf_inputs = [
            (embedding_search_list, 1.0),
        ]

        # 3. RRF 合并
        rrf_merge_results = self._rrf_merge(rrf_inputs, max_results=self.MAX_RESULTS)

        # 4. 仅提取文档 (不含分数)
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        logger.info("RRF fusion: %d input -> %d output", len(embedding_search_list), len(rrf_chunks))
        return {"rrf_chunks": rrf_chunks}

    def _rrf_merge(
        self,
        rrf_inputs: List[Tuple[List[Dict], float]],
        max_results: int = None,
    ) -> List[Tuple[Dict, float]]:
        """
        倒数排名融合算法。

        对来自所有检索路径的每个文档计算:
            score = sum(weight * 1 / (k + rank))

        然后按总分降序排序, 取前 max_results 个。

        Args:
            rrf_inputs: (doc_list, weight) 元组列表。
            max_results: 返回的最大文档数 (None = 全部)。

        Returns:
            按分数降序排列的 (doc, score) 元组列表。
        """
        # 按 chunk_id 累加分数
        chunk_scores: Dict[Any, float] = {}
        chunk_data: Dict[Any, Dict] = {}

        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get("chunk_id")
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (self.K + rank)
                # 仅在首次出现时记录文档
                chunk_data.setdefault(chunk_id, doc)

        # 按分数降序排序
        unsorted_results = [
            (chunk_data[cid], score) for cid, score in chunk_scores.items()
        ]
        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        # 截断
        if max_results:
            return sorted_results[:max_results]
        return sorted_results
