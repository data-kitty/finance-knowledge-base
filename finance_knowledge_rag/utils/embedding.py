"""
BGE-M3 嵌入工具模块。

提供 BGEM3EmbeddingFunction 单例及批量稠密+稀疏向量生成。
所有配置 (模型路径、设备、fp16) 均从 rag.config 读取。
"""
import logging
from typing import List, Dict, Optional

from finance_knowledge_rag.config import rag_config

logger = logging.getLogger(__name__)

# 单例嵌入函数
_bge_m3_ef = None


def get_bge_m3_ef():
    """
    获取 BGEM3EmbeddingFunction 单例实例 (懒初始化)。

    从配置指定的路径加载 BGE-M3 模型。

    Returns:
        BGEM3EmbeddingFunction 实例。
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    model_name = rag_config.bge_m3.bge_m3_path
    device = rag_config.bge_m3.bge_device
    use_fp16 = rag_config.bge_m3.bge_fp16

    if not model_name:
        raise ValueError("BGE_M3_PATH is not configured")

    logger.info("Loading BGE-M3 model from '%s' (device=%s, fp16=%s)", model_name, device, use_fp16)
    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
    )
    logger.info("BGE-M3 model loaded successfully")
    return _bge_m3_ef


def generate_embeddings(texts: List[str]) -> Dict[str, list]:
    """
    为文本列表生成 BGE-M3 稠密 + 稀疏向量。

    Args:
        texts: 要向量化的文本字符串列表。

    Returns:
        包含以下键的 Dict:
            "dense":  List[List[float]]  — 稠密向量
            "sparse": List[Dict[int, float]] — 稀疏向量, 形式为 {token_id: weight}

    Raises:
        ValueError: 当 texts 为空时。
    """
    if not texts:
        raise ValueError("texts must not be empty")

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)

    # 将稀疏 CSR 矩阵转换为字典列表
    processed_sparse: List[Dict[int, float]] = []
    for i in range(len(texts)):
        start = embeddings["sparse"].indptr[i]
        end = embeddings["sparse"].indptr[i + 1]
        indices = embeddings["sparse"].indices[start:end].tolist()
        data = embeddings["sparse"].data[start:end].tolist()
        sparse_dict = {k: v for k, v in zip(indices, data)}
        processed_sparse.append(sparse_dict)

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse,
    }
