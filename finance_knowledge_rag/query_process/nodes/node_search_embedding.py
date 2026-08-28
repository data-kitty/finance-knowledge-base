"""
查询工作流的向量检索节点。

为改写后的查询生成 BGE-M3 向量, 然后在 chunks_collection 上
执行混合检索 (稠密 + 稀疏), 并按 item_names 过滤。
"""
import logging
from typing import Any, Dict, Tuple

from finance_knowledge_rag.config import rag_config
from finance_knowledge_rag.query_process.base import NodeBase
from finance_knowledge_rag.query_process.state import QueryGraphState
from finance_knowledge_rag.utils.embedding import generate_embeddings
from finance_knowledge_rag.utils.milvus_utils import (
    escape_milvus_string,
    create_hybrid_search_request,
    hybrid_search,
)

logger = logging.getLogger(__name__)


class NodeSearchEmbedding(NodeBase):
    """
    向量检索节点: 对 chunks_collection 执行混合检索。

    使用 rewritten_query 生成向量, 用 item_names 进行标量过滤,
    将检索范围缩小到已确认的金融主题。
    """

    name = "node_search_embedding"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        执行混合向量检索。

        Args:
            state: 必须包含 'rewritten_query' 和 'item_names'。

        Returns:
            包含 'embedding_chunks' 的字典 — 检索结果命中列表。
        """
        try:
            # 步骤 1: 校验参数
            rewritten_query, item_names = self._step1_validate_param(state)

            # 步骤 2: 向量检索
            res = self._step2_search_embedding(
                rewritten_query=rewritten_query,
                item_names=item_names,
            )

            return {"embedding_chunks": res}

        except Exception as e:
            logger.exception("Vector search failed: %s", e)
            return {"embedding_chunks": []}

    def _step1_validate_param(self, state: QueryGraphState) -> Tuple[str, list]:
        """校验 rewritten_query 和 item_names。"""
        rewritten_query = state.get("rewritten_query", "")
        if not rewritten_query:
            raise ValueError("rewritten_query is required")

        item_names = state.get("item_names", [])
        if not item_names:
            raise ValueError("item_names is required")

        return rewritten_query, item_names

    def _step2_search_embedding(self, rewritten_query: str, item_names: list):
        """生成向量并执行混合检索。"""
        try:
            # 1. 对改写后的查询向量化
            embeddings = generate_embeddings([rewritten_query])
            dense_vector = embeddings.get("dense", [])[0]
            sparse_vector = embeddings.get("sparse", [])[0]

            # 2. 构建标量过滤表达式
            expr = None
            if item_names:
                escaped = ", ".join(
                    f'"{escape_milvus_string(name)}"' for name in item_names
                )
                expr = f"item_name in [{escaped}]"
            else:
                logger.info("No item_names specified, searching full collection")

            # 3. 构建混合检索请求
            reqs = create_hybrid_search_request(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=10,
            )

            # 4. 执行混合检索
            res = hybrid_search(
                collection_name=rag_config.milvus.chunks_collection,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                output_fields=["chunk_id", "content", "item_name", "title"],
                limit=10,
            )

            return res[0] if res else []

        except Exception as e:
            logger.exception("Hybrid search execution failed: %s", e)
            raise
