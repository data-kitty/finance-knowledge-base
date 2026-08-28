"""
导入工作流的主题 (金融主题) 提取节点。

使用 LLM 从文件标题和前 5 个切片中识别金融主题名称
(例如 "个人贷款", "理财", "信用卡")。将 item_name 回填到所有切片。
为主题名称生成向量, 并存入 Milvus 的 item_name_collection。
"""
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from finance_knowledge_rag.config import rag_config
from finance_knowledge_rag.import_process.base import NodeBase
from finance_knowledge_rag.import_process.prompt import NAME_RECOGNITION, SYSTEM_PROMPT
from finance_knowledge_rag.import_process.state import ImportGraphState
from finance_knowledge_rag.utils.embedding import generate_embeddings
from finance_knowledge_rag.utils.llm import get_llm_client
from finance_knowledge_rag.utils.milvus_utils import get_milvus_client

logger = logging.getLogger(__name__)


class NodeItemExtract(NodeBase):
    """
    金融主题提取节点。

    流程:
    1. 从 state 提取 file_title 和 chunks
    2. 用前 5 个切片构建 LLM 上下文
    3. 调用 LLM 识别金融主题名称
    4. 将 item_name 回填到所有切片
    5. 为主题名称生成稠密+稀疏向量
    6. 将主题名称 + 向量持久化到 Milvus item_name_collection
    """

    name = "node_item_extract"

    # 使用前 5 个切片作为 LLM 上下文
    DEFAULT_ITEM_NAME_CHUNK_K = 5
    # 上下文最大总字符数
    MAX_CHARS = 2500

    def process(self, state: ImportGraphState) -> Dict[str, Any]:
        """执行金融主题提取流水线。"""
        # 步骤 1: 校验输入
        file_title, chunks = self._step1_get_inputs(state)

        # 步骤 2: 构建 LLM 上下文
        context = self._step2_build_context(chunks)

        # 步骤 3: 调用 LLM 识别主题名称
        item_name = self._step3_call_llm(file_title, context)

        # 步骤 4: 将 item_name 回填到所有切片
        chunks = self._step4_update_chunks(chunks, item_name)

        # 步骤 5: 为主题名称生成向量
        dense_vector, sparse_vector = self._step5_generate_vectors(item_name)

        # 步骤 6: 持久化到 Milvus item_name_collection
        self._step6_save_to_milvus(file_title, item_name, dense_vector, sparse_vector)

        logger.info("Finance topic recognized: '%s' (file: %s)", item_name, file_title)

        return {
            "chunks": chunks,
            "item_name": item_name,
        }

    def _step1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        """校验并提取 file_title 和 chunks。"""
        file_title = state.get("file_title")
        if not file_title:
            raise ValueError("file_title must not be empty")

        chunks = state.get("chunks")
        if not chunks:
            raise ValueError("chunks must not be empty")
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")

        return file_title, chunks

    def _step2_build_context(self, chunks: List[Dict]) -> str:
        """
        用前 K 个切片构建格式化上下文字符串。

        将总字符数限制在 MAX_CHARS 以内, 提高 LLM 输入效率。
        """
        total_chars = 0
        parts: List[str] = []

        for idx, chunk in enumerate(chunks[:self.DEFAULT_ITEM_NAME_CHUNK_K], start=1):
            chunk_title = chunk.get("title", "")
            chunk_content = chunk.get("content", "")
            piece = f"[Chunk {idx}]\nTitle: {chunk_title}\nContent: {chunk_content}"
            parts.append(piece)
            total_chars += len(piece)

            if total_chars > self.MAX_CHARS:
                logger.warning("Context exceeded %d chars, truncated at chunk %d", self.MAX_CHARS, idx)
                break

        context = "\n\n".join(parts).strip()
        return context[:self.MAX_CHARS]

    def _step3_call_llm(self, file_title: str, context: str) -> str:
        """调用 LLM 识别金融主题名称。"""
        try:
            prompt = NAME_RECOGNITION.format(
                file_title=file_title,
                context=context,
            )

            llm = get_llm_client(model=rag_config.llm.item_model)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)

            item_name = response.content.strip()
            # 清理空白字符
            item_name = re.sub(r"\s+", "", item_name)

            if not item_name:
                logger.warning("LLM returned empty topic name, using file_title as fallback")
                return file_title

            return item_name

        except Exception as e:
            logger.exception("LLM topic recognition failed: %s", e)
            return file_title

    def _step4_update_chunks(
        self, chunks: List[Dict], item_name: str
    ) -> List[Dict]:
        """将 item_name 回填到所有切片。"""
        for chunk in chunks:
            chunk["item_name"] = item_name
        return chunks

    def _step5_generate_vectors(self, item_name: str):
        """为主题名称生成 BGE-M3 稠密 + 稀疏向量。"""
        vectors = generate_embeddings([item_name])
        return vectors["dense"][0], vectors["sparse"][0]

    def _step6_save_to_milvus(
        self,
        file_title: str,
        item_name: str,
        dense_vector,
        sparse_vector,
    ) -> None:
        """
        将主题名称 + 向量持久化到 Milvus item_name_collection。

        若集合不存在则创建。插入前执行幂等删除:
        移除相同 file_title 的旧数据。
        """
        try:
            client = get_milvus_client()
            collection_name = rag_config.milvus.item_name_collection

            # 若集合不存在则创建
            if not client.has_collection(collection_name):
                self._create_item_name_collection(collection_name, client)

            # 幂等: 删除相同 file_title 的旧数据
            client.delete(
                collection_name=collection_name,
                filter=f"file_title=='{file_title}'",
            )

            # 插入新数据
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
            client.insert(collection_name=collection_name, data=[data])
            logger.info(
                "Saved topic '%s' to collection '%s'", item_name, collection_name
            )

        except Exception as e:
            logger.exception(
                "Failed to persist topic to Milvus (file=%s): %s", file_title, e
            )
            raise

    def _create_item_name_collection(self, collection_name: str, client) -> None:
        """创建带 schema 和索引的 item_name 集合。"""
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 主键 (INT64, 自增)
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        # 文件标题
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        # 主题 (金融主题) 名称
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        # 稠密向量 (BGE-M3, 1024 维)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024,
        )
        # 稀疏向量 (BGE-M3)
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
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
        logger.info("Created item_name collection: %s", collection_name)
