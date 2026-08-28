"""
Milvus 工具模块。

提供 Milvus 客户端单例、字符串转义、混合检索请求组装
与混合检索执行。所有配置均从 rag.config 读取。
"""
import logging
from typing import List, Optional

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from finance_knowledge_rag.config import rag_config

logger = logging.getLogger(__name__)

# Milvus 客户端单例
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """
    获取 MilvusClient 单例实例 (懒初始化)。

    Returns:
        MilvusClient 实例。

    Raises:
        ValueError: 当 MILVUS_URL 未配置时。
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    url = rag_config.milvus.milvus_url
    if not url:
        raise ValueError("MILVUS_URL is not configured")

    _milvus_client = MilvusClient(uri=url)
    logger.info("Milvus client connected to %s", url)
    return _milvus_client


def escape_milvus_string(value: str) -> str:
    """
    转义字符串中的特殊字符, 以便安全用于 Milvus 过滤表达式。

    转义反斜杠、双引号和单引号, 防止过滤解析错误。

    Args:
        value: 待转义的原始字符串。

    Returns:
        转义后的安全字符串。
    """
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")


def create_hybrid_search_request(
    dense_vector: list,
    sparse_vector: dict,
    dense_params: Optional[dict] = None,
    sparse_params: Optional[dict] = None,
    expr: Optional[str] = None,
    limit: int = 5,
) -> List[AnnSearchRequest]:
    """
    组装混合 (稠密 + 稀疏) 检索请求列表。

    Args:
        dense_vector: 稠密嵌入向量。
        sparse_vector: 稀疏嵌入字典 {token_id: weight}。
        dense_params: 稠密检索的 Milvus 搜索参数 (默认 COSINE)。
        sparse_params: 稀疏检索的 Milvus 搜索参数 (默认 IP)。
        expr: 标量过滤表达式 (可选)。
        limit: 每个子检索的 Top-K 限制。

    Returns:
        两个 AnnSearchRequest 对象的列表 [dense, sparse]。
    """
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}

    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    request_dense = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )

    request_sparse = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )

    return [request_dense, request_sparse]


def hybrid_search(
    collection_name: str,
    reqs: List[AnnSearchRequest],
    ranker_weights: tuple = (0.5, 0.5),
    norm_score: bool = False,
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
) -> list:
    """
    在指定的 Milvus 集合上执行混合向量检索。

    Args:
        collection_name: 目标 Milvus 集合。
        reqs: AnnSearchRequest 列表 (稠密 + 稀疏)。
        ranker_weights: WeightedRanker 的权重 (稠密权重, 稀疏权重)。
        norm_score: 兼容保留参数。pymilvus 2.4.9 的 WeightedRanker 已移除
                    norm_score(只接受权重位置参数), 分数归一化由混合检索内部处理,
                    故本参数不再生效。
        limit: 最终 Top-K 限制。
        output_fields: 结果中需要返回的字段。

    Returns:
        检索结果命中列表。

    Raises:
        RuntimeError: 当混合检索失败时。
    """
    try:
        # pymilvus 2.4.9: WeightedRanker.__init__(*nums), 旧版 norm_score 参数已移除
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1])
        client = get_milvus_client()

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
        )

        logger.info("Hybrid search completed on '%s'", collection_name)
        return res
    except Exception as e:
        logger.exception("Hybrid search failed on '%s': %s", collection_name, e)
        raise RuntimeError(f"Hybrid search failed: {e}")
