"""
RAG 知识库服务路由。

对外提供 HTTP 接口, 供客服后端等下游服务调用:

  - POST /api/rag/query    知识问答(同步返回答案 + 命中主题)
  - POST /api/rag/import   重新导入知识文档(扫描 knowledge_docs 下全部 .md)
  - GET  /api/rag/collections  Milvus 集合状态(排障用)
  - GET  /health           健康检查

约定:
  - 查询失败不抛 500, 返回 answer="" 与 error 字段, 由调用方降级到 FAQ;
  - 导入失败返回 500 与错误详情(导入是管理操作, 失败需要明确暴露)。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from finance_knowledge_rag.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])

# RAG 服务单例: 构造不连接任何外部资源(工作流懒加载), 全局复用
_rag_service = RAGService()


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """知识问答请求。"""

    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID(用于历史记忆), 缺省自动生成")


class QueryResponse(BaseModel):
    """知识问答响应。"""

    answer: str = Field("", description="答案文本; 查询失败为空字符串")
    session_id: str = Field("", description="实际使用的会话 ID")
    item_names: List[str] = Field(default_factory=list, description="命中的金融主题")
    rewritten_query: str = Field("", description="改写后的问题(诊断用)")
    error: str = Field("", description="查询失败时的错误信息(成功为空)")


class ImportRequest(BaseModel):
    """文档导入请求。"""

    docs_dir: Optional[str] = Field(None, description="知识文档目录, 缺省用 .env 的 KNOWLEDGE_DOCS_DIR")


class ImportResponse(BaseModel):
    """文档导入响应。"""

    results: List[Dict[str, Any]] = Field(default_factory=list, description="每个文件的导入结果摘要")
    error: str = Field("", description="导入失败时的错误信息(成功为空)")


class CollectionInfo(BaseModel):
    """Milvus 集合状态。"""

    name: str
    row_count: Optional[int] = None


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, summary="知识问答")
async def query_rag(request: QueryRequest) -> QueryResponse:
    """运行 RAG 查询工作流(意图确认 → 混合检索 → RRF → 重排 → 答案生成)。"""
    answer, state = await _query_with_state(request.question, request.session_id)
    return QueryResponse(
        answer=answer,
        session_id=state.get("session_id") or "",
        item_names=list(state.get("item_names") or []),
        rewritten_query=state.get("rewritten_query") or "",
    )


@router.post("/import", response_model=ImportResponse, summary="重新导入知识文档")
async def import_documents(request: ImportRequest) -> ImportResponse:
    """扫描知识文档目录, 逐个文件执行导入工作流(幂等, 按 file_title 清理旧数据)。"""
    try:
        results = await _rag_service.import_documents(docs_dir=request.docs_dir)
        return ImportResponse(results=results)
    except Exception as e:
        logger.error("RAG import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG 文档导入失败: {e}")


@router.get("/collections", response_model=List[CollectionInfo], summary="Milvus 集合状态")
async def list_collections() -> List[CollectionInfo]:
    """返回 Milvus 中 RAG 相关集合的名称与行数(排障用)。"""
    try:
        from finance_knowledge_rag.utils.milvus_utils import get_milvus_client

        client = get_milvus_client()
        names = client.list_collections()
        items: List[CollectionInfo] = []
        for name in names:
            count: Optional[int] = None
            try:
                count = client.query(
                    collection_name=name,
                    filter="",
                    output_fields=["count(*)"],
                )[0].get("count(*)")
            except Exception:
                count = None
            items.append(CollectionInfo(name=name, row_count=count))
        return items
    except Exception as e:
        logger.error("List collections failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Milvus 连接失败: {e}")


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------

async def _query_with_state(question: str, session_id: Optional[str]) -> tuple[str, Dict[str, Any]]:
    """执行查询并返回 (answer, 最终图状态); 失败时 answer 为空、state 含 error。"""
    answer, state = await _rag_service.query_with_state(question, session_id)
    return answer, state
