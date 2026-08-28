"""
导入工作流的 BGE-M3 向量化节点。

为所有切片批量生成稠密 + 稀疏向量。以 3 为批次处理
以避免 OOM。每个切片的文本由 item_name 前缀 + 内容组成,
以获得更丰富的语义。
"""
import logging
from typing import Any, Dict, List

from finance_knowledge_rag.import_process.base import NodeBase
from finance_knowledge_rag.import_process.state import ImportGraphState
from finance_knowledge_rag.utils.embedding import generate_embeddings

logger = logging.getLogger(__name__)


class NodeEmbedding(NodeBase):
    """
    BGE-M3 向量化节点: 将切片文本转换为稠密 + 稀疏向量。

    按批次处理切片 (batch_size=3) 以控制内存。在内容前
    拼接 item_name 以增强语义匹配。
    """

    name = "node_embedding"

    # 向量化生成的批大小
    BATCH_SIZE = 3

    def process(self, state: ImportGraphState) -> Dict[str, Any]:
        """
        为所有切片批量生成向量。

        Args:
            state: 必须包含带 'item_name' 和 'content' 字段的 'chunks'。

        Returns:
            包含 'chunks' 的字典 (每个切片现在带 dense_vector + sparse_vector)。
        """
        # 步骤 1: 校验输入
        chunks = self._step1_validate_input(state)

        # 步骤 2: 分批生成向量
        output_data = self._step2_generate_embeddings(chunks)

        return {"chunks": output_data}

    def _step1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """校验 chunks 存在且为非空列表。"""
        chunks = state.get("chunks")
        if not chunks:
            raise ValueError("chunks must not be empty")
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        return chunks

    def _step2_generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        分批生成向量。

        每个切片的文本为 "{item_name}\n{content}", 以增强
        与金融主题名称的语义匹配。
        """
        output_data: List[Dict] = []
        batch_size = self.BATCH_SIZE

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # 组合文本: item_name + content
            texts = [
                f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}"
                for chunk in batch_chunks
            ]

            # 批量生成向量
            vectors = generate_embeddings(texts)
            dense_vectors = vectors.get("dense", [])
            sparse_vectors = vectors.get("sparse", [])

            # 将向量写回切片
            for j, chunk in enumerate(batch_chunks):
                chunk["dense_vector"] = dense_vectors[j]
                chunk["sparse_vector"] = sparse_vectors[j]
                output_data.append(chunk)

            logger.info(
                "Embedded batch %d-%d / %d", i + 1, min(i + batch_size, len(chunks)), len(chunks)
            )

        logger.info("Embedding complete: %d chunks", len(output_data))
        return output_data
