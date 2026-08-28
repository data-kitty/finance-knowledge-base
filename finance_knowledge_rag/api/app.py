"""
RAG 知识库服务 — FastAPI 应用工厂。

独立服务, 不依赖客服后端; 通过 HTTP 向客服后端等下游提供
知识问答 / 文档导入能力。RAGService 内部懒加载, 启动不连接
Milvus / 不加载 BGE-M3 模型, 首次查询时才初始化工作流。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finance_knowledge_rag.api.routers.rag_router import router as rag_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭日志。RAGService 懒加载, 无需额外初始化。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Finance Knowledge RAG Service...")
    yield
    logger.info("Finance Knowledge RAG Service stopped.")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="Finance Knowledge RAG API",
        description="金融知识库 RAG 服务(文档导入 + 知识问答), 供客服后端等下游调用",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(rag_router)

    @app.get("/health", tags=["health"])
    async def health():
        """健康检查。"""
        return {"status": "ok"}

    return app


app = create_app()
