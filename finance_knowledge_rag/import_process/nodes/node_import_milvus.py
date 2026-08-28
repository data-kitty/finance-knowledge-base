"""
导入工作流的 Milvus 入库节点。

将带向量的切片持久化到 Milvus chunks_collection。若集合不存在,
则自动创建 schema + 索引。执行幂等删除: 移除相同 file_title 的
旧数据。批量插入并回填自动生成的 chunk_id。
"""
import logging
from typing import Any, Dict, List

from pymilvus import DataType

from finance_knowledge_rag.config import rag_config
from finance_knowledge_rag.import_process.base import NodeBase
from finance_knowledge_rag.import_process.state import ImportGraphState
from finance_knowledge_rag.utils.milvus_utils import get_milvus_client, escape_milvus_string

logger = logging.getLogger(__name__)


class NodeImportMilvus(NodeBase):
    """
    Milvus 入库节点: 将带向量的切片持久化到 Milvus。

    Schema 字段: chunk_id (pk), content, title, parent_title, part,
    file_title, item_name, sparse_vector, dense_vector。

    索引: dense_vector -> AUTOINDEX+COSINE, sparse_vector -> SPARSE_INVERTED_INDEX+IP。
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState) -> Dict[str, Any]:
        """
        执行 Milvus 入库流水线。

        Args:
            state: 必须包含 'file_title' 和 'chunks' (带向量)。

        Returns:
            包含 'chunks' 的字典 — 切片已回填 chunk_id。
        """
        # 步骤 1: 校验输入
        file_title, chunks_data, vector_dim = self._step1_check_input(state)

        # 步骤 2: 准备集合 (必要时自动创建)
        self._step2_prepare_collection(vector_dim)

        # 步骤 3: 幂等清理——删除相同 file_title 的旧数据
        self._step3_clean_old_data(file_title)

        # 步骤 4: 批量插入 + 回填 chunk_id
        updated_chunks = self._step4_insert_data(chunks_data)

        return {"chunks": updated_chunks}

    def _step1_check_input(self, state: Dict[str, Any]) -> tuple:
        """校验 file_title 和 chunks, 提取向量维度。"""
        file_title = state.get("file_title")
        if not file_title:
            raise ValueError("file_title must not be empty")

        chunks = state.get("chunks")
        if not chunks:
            raise ValueError("chunks must not be empty")
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")

        # 校验向量存在
        first_chunk = chunks[0]
        if "dense_vector" not in first_chunk:
            raise ValueError("chunks missing 'dense_vector' field")
        if "sparse_vector" not in first_chunk:
            raise ValueError("chunks missing 'sparse_vector' field")

        vector_dim = len(first_chunk["dense_vector"])
        return file_title, chunks, vector_dim

    def _step2_prepare_collection(self, vector_dim: int) -> None:
        """确保 chunks 集合存在, 必要时创建。"""
        client = get_milvus_client()
        collection_name = rag_config.milvus.chunks_collection

        if not client.has_collection(collection_name):
            self._create_chunks_collection(collection_name, client, vector_dim)
            logger.info("Created chunks collection: %s", collection_name)

    def _create_chunks_collection(self, collection_name: str, client, vector_dim: int) -> None:
        """创建带完整 schema 和索引的 chunks 集合。"""
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 主键——自动生成的 INT64
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        # 切片内容 (VARCHAR, 最大 65535)
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        # 切片标题
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        # 父标题 (用于切分的节)
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        # 部分编号 (用于切分的节)
        schema.add_field(
            field_name="part",
            datatype=DataType.INT8,
        )
        # 源文件标题
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        # 金融主题名称 (item_name)
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        # 稀疏向量 (BGE-M3)
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        # 稠密向量 (BGE-M3)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_dim,
        )

        # 构建索引
        index_params = client.prepare_index_params()
        # 稠密: AUTOINDEX + COSINE
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        # 稀疏: SPARSE_INVERTED_INDEX + IP
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "normalize": True,
                "quantization": "none",
            },
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _step3_clean_old_data(self, file_title: str) -> None:
        """幂等: 删除相同 file_title 的旧切片。"""
        safe_file_title = escape_milvus_string(file_title)
        filter_expr = f'file_title=="{safe_file_title}"'

        client = get_milvus_client()
        collection_name = rag_config.milvus.chunks_collection
        client.delete(collection_name=collection_name, filter=filter_expr)
        logger.info("Cleaned old data for file_title='%s'", file_title)

    def _step4_insert_data(self, chunks_data: List[Dict]) -> List[Dict]:
        """批量插入切片到 Milvus 并回填 chunk_id。"""
        client = get_milvus_client()
        collection_name = rag_config.milvus.chunks_collection

        result = client.insert(
            collection_name=collection_name,
            data=chunks_data,
        )

        # 回填 chunk_id
        inserted_ids = result.get("ids", [])
        for idx, chunk in enumerate(chunks_data):
            if idx < len(inserted_ids):
                chunk["chunk_id"] = inserted_ids[idx]

        logger.info("Inserted %d chunks to '%s'", len(chunks_data), collection_name)
        return chunks_data
