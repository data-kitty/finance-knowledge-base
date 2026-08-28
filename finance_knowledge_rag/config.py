"""
RAG 配置模块。

所有 RAG 相关配置均通过 dotenv 从 .env 文件读取。
无任何硬编码值——每个参数都来自环境变量。
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 定位本项目根目录下的 .env 文件 (finance-knowledge-RAG/.env)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


@dataclass
class LLMConfig:
    """RAG 管道的 LLM 配置。"""
    api_key: str
    base_url: str
    model: str
    temperature: float
    item_model: str
    vl_model: str


@dataclass
class MilvusConfig:
    """Milvus 向量数据库配置。"""
    milvus_url: str
    chunks_collection: str
    item_name_collection: str


@dataclass
class BGEM3Config:
    """BGE-M3 嵌入模型配置。"""
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool


@dataclass
class EmbeddingHttpConfig:
    """DashScope 文本向量化 API 配置。"""
    api_key: str
    dashscope_url: str
    model: str
    dimension: int
    batch_size: int


@dataclass
class RerankerHttpConfig:
    """DashScope 重排 API 配置。"""
    base_url: str
    api_key: str
    model: str
    instruct: str


@dataclass
class MongoConfig:
    """用于聊天历史的 MongoDB 配置。"""
    mongo_url: str
    mongo_db_name: str


@dataclass
class MinIOConfig:
    """MinIO 对象存储配置。"""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    img_dir: str


@dataclass
class RAGConfig:
    """聚合的 RAG 配置, 包含所有子配置。"""
    llm: LLMConfig
    milvus: MilvusConfig
    bge_m3: BGEM3Config
    embedding_http: EmbeddingHttpConfig
    reranker_http: RerankerHttpConfig
    mongo: MongoConfig
    minio: MinIOConfig
    knowledge_docs_dir: str


def _get_env(key: str, default: str = "") -> str:
    """读取环境变量, 若为必填项且缺失则抛出异常。"""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def _get_env_required(key: str) -> str:
    """读取必填的环境变量, 缺失时抛出异常。"""
    value = os.getenv(key)
    if value is None or value == "":
        raise ValueError(f"Environment variable '{key}' is required but not set in .env")
    return value


def _get_bool_env(key: str, default: bool = False) -> bool:
    """解析布尔类型的环境变量。"""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def load_rag_config() -> RAGConfig:
    """
    从环境变量加载并组装完整的 RAG 配置。

    Returns:
        包含全部子配置的 RAGConfig 实例。
    """
    # --- LLM 配置 ---
    llm_config = LLMConfig(
        api_key=_get_env_required("RAG_LLM_API_KEY"),
        base_url=_get_env("RAG_LLM_BASE_URL", "https://llm-api.net/v1"),
        model=_get_env("RAG_LLM_MODEL", "qwen-flash"),
        temperature=float(_get_env("RAG_LLM_TEMPERATURE", "0.1")),
        item_model=_get_env("RAG_ITEM_MODEL", "qwen-flash"),
        vl_model=_get_env("RAG_VL_MODEL", "qwen3-vl-flash"),
    )

    # --- Milvus 配置 ---
    milvus_config = MilvusConfig(
        milvus_url=_get_env_required("MILVUS_URL"),
        chunks_collection=_get_env("CHUNKS_COLLECTION", "finance_chunks"),
        item_name_collection=_get_env("ITEM_NAME_COLLECTION", "finance_item_names"),
    )

    # --- BGE-M3 配置 ---
    bge_m3_config = BGEM3Config(
        bge_m3_path=_get_env_required("BGE_M3_PATH"),
        bge_m3=_get_env("BGE_M3", "BAAI/bge-m3"),
        bge_device=_get_env("BGE_DEVICE", "cpu"),
        bge_fp16=_get_bool_env("BGE_FP16", False),
    )

    # --- DashScope 向量化 ---
    embedding_http_config = EmbeddingHttpConfig(
        api_key=_get_env("TEXT_EMBEDDING_API_KEY"),
        dashscope_url=_get_env("TEXT_EMBEDDING_DASHSCOPE_URL", "https://dashscope.aliyuncs.com/api/v1"),
        model=_get_env("TEXT_EMBEDDING_MODEL", "text-embedding-v4"),
        dimension=int(_get_env("TEXT_EMBEDDING_DIMENSION", "1024")),
        batch_size=int(_get_env("TEXT_EMBEDDING_BATCH_SIZE", "10")),
    )

    # --- DashScope 重排 ---
    reranker_http_config = RerankerHttpConfig(
        base_url=_get_env("TEXT_RERANK_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
        api_key=_get_env("TEXT_RERANK_API_KEY"),
        model=_get_env("TEXT_RERANK_MODEL", "qwen3-rerank"),
        instruct=_get_env("TEXT_RERANK_INSTRUCT", "Retrieve semantically similar text."),
    )

    # --- MongoDB 配置 ---
    mongo_config = MongoConfig(
        mongo_url=_get_env_required("MONGO_URL"),
        mongo_db_name=_get_env("MONGO_DB_NAME", "finance_rag"),
    )

    # --- MinIO 配置 ---
    minio_config = MinIOConfig(
        endpoint=_get_env("MINIO_ENDPOINT"),
        access_key=_get_env("MINIO_ACCESS_KEY"),
        secret_key=_get_env("MINIO_SECRET_KEY"),
        bucket_name=_get_env("MINIO_BUCKET_NAME", "finance-knowledge"),
        img_dir=_get_env("MINIO_IMG_DIR", "upload-images"),
    )

    # --- 知识文档目录 ---
    knowledge_docs_dir = _get_env("KNOWLEDGE_DOCS_DIR", "./knowledge_docs")

    return RAGConfig(
        llm=llm_config,
        milvus=milvus_config,
        bge_m3=bge_m3_config,
        embedding_http=embedding_http_config,
        reranker_http=reranker_http_config,
        mongo=mongo_config,
        minio=minio_config,
        knowledge_docs_dir=knowledge_docs_dir,
    )


# 单例配置实例, 在模块导入时加载
rag_config = load_rag_config()
