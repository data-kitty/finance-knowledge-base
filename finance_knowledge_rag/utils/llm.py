"""
LLM 客户端工具模块。

提供带缓存的 ChatOpenAI 客户端工厂。所有配置 (api_key, base_url,
model, temperature) 均从 rag.config 读取。已禁用思考模式。
"""
import logging
from typing import Optional, Dict

from langchain_openai import ChatOpenAI

from finance_knowledge_rag.config import rag_config

logger = logging.getLogger(__name__)

# 按 (model, json_mode) 键控的缓存, 用于单例访问
_llm_client_cache: Dict[tuple, ChatOpenAI] = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取指定模型的 ChatOpenAI 缓存实例。

    禁用思考模式 (enable_thinking) 以保证输出确定性。

    Args:
        model: 模型名称; 若为 None, 使用配置中的默认值 (RAG_LLM_MODEL)。
        json_mode: 若为 True, 将 response_format 设为 json_object。

    Returns:
        ChatOpenAI 单例实例。
    """
    m = model or rag_config.llm.model
    key = (m, json_mode)

    if key in _llm_client_cache:
        return _llm_client_cache[key]

    # 禁用思考模式——不同模型默认行为不同, 此处统一关闭
    extra_body = {"enable_thinking": False}

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    llm = ChatOpenAI(
        model=m,
        api_key=rag_config.llm.api_key,
        base_url=rag_config.llm.base_url,
        temperature=rag_config.llm.temperature,
        extra_body=extra_body,
        model_kwargs=model_kwargs,
    )

    _llm_client_cache[key] = llm
    logger.info("LLM client created: model=%s, json_mode=%s", m, json_mode)
    return llm
