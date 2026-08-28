"""
RAGService — RAG 知识库对外服务接口.

职责:
  - import_documents(docs_dir): 扫描 knowledge_docs 下全部 .md, 逐个跑导入工作流
    (切片 → 主题标签 → BGE-M3 向量化 → Milvus 入库)
  - query(question, session_id): 运行查询工作流(意图确认 → 混合检索 → RRF
    → 重排 → 答案生成), 返回最终答案文本
  - query_stream(): 流式查询入口(当前实现为一次性返回完整答案,
    上层 SSE 层负责分块下发)

内部懒加载: 工作流对象在首次使用时创建, 避免启动时连接 Milvus / 加载模型.
查询工作流为同步 LangGraph 执行, 通过 asyncio.to_thread 放到线程池,
避免阻塞 FastAPI 事件循环.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from finance_knowledge_rag.config import rag_config
from finance_knowledge_rag.import_process.main_graph import KBImportWorkflow
from finance_knowledge_rag.query_process.main_graph import KBQueryWorkflow

__all__ = ["RAGService"]

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 知识库服务: 文档导入 + 知识问答."""

    def __init__(self) -> None:
        self._query_workflow: Optional[KBQueryWorkflow] = None

    # ------------------------------------------------------------------
    # 文档导入
    # ------------------------------------------------------------------

    async def import_documents(self, docs_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """扫描目录下所有 .md 文件并执行导入工作流.

        Args:
            docs_dir: 知识文档目录; 缺省用 rag_config.knowledge_docs_dir.

        Returns:
            每个文件的导入结果摘要列表(文件名 + 切片数).
        """
        docs_path = Path(docs_dir or rag_config.knowledge_docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Knowledge docs directory not found: {docs_path}")

        md_files = sorted(docs_path.glob("*.md"))
        if not md_files:
            logger.warning("No .md files found in %s", docs_path)
            return []

        results: List[Dict[str, Any]] = []
        for file_path in md_files:
            init_state: Dict[str, Any] = {
                "task_id": f"import_{file_path.stem}",
                "local_file_path": str(file_path),
            }
            # 导入工作流同步执行, 放线程池避免阻塞事件循环
            final_state = await asyncio.to_thread(self._run_import_workflow, init_state)
            results.append(
                {
                    "file_title": file_path.stem,
                    "chunk_count": len(final_state.get("chunks") or []),
                    "item_name": final_state.get("item_name", ""),
                }
            )
            logger.info(
                "Imported '%s': %d chunks, item_name='%s'",
                file_path.stem,
                len(final_state.get("chunks") or []),
                final_state.get("item_name", ""),
            )

        return results

    # ------------------------------------------------------------------
    # 知识问答
    # ------------------------------------------------------------------

    async def query(self, question: str, session_id: Optional[str] = None) -> str:
        """查询知识库并返回答案.

        Args:
            question: 用户问题.
            session_id: 会话 ID(用于历史记忆); 缺省生成新的.

        Returns:
            答案文本; 查询失败时返回空字符串(上层降级到 FAQ).
        """
        answer, _ = await self.query_with_state(question, session_id)
        return answer

    async def query_with_state(
        self, question: str, session_id: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """查询知识库并返回 (答案, 最终图状态).

        图状态含 item_names / rewritten_query 等诊断字段,
        供 HTTP 层返回给调用方做可观测性展示.

        Returns:
            (answer, final_state); 查询失败时 answer 为空字符串.
        """
        if not question or not question.strip():
            return "", {"session_id": session_id or ""}

        sid = session_id or f"rag-{uuid.uuid4().hex[:12]}"
        init_state: Dict[str, Any] = {
            "session_id": sid,
            "original_query": question.strip(),
            "task_id": f"query_{uuid.uuid4().hex[:8]}",
            "is_stream": False,
        }

        try:
            final_state = await asyncio.to_thread(
                self._run_query_workflow, init_state
            )
            return final_state.get("answer", "") or "", final_state
        except Exception as e:
            logger.error("RAG query failed: %s", e, exc_info=True)
            return "", {"session_id": sid, "error": str(e)}

    async def query_stream(self, question: str, session_id: Optional[str] = None):
        """流式查询入口.

        当前实现: 内部走完整查询返回答案文本, 以单个 chunk 产出,
        SSE 层负责按固定大小分块下发(与 chat_router 的分块策略一致).
        """
        answer = await self.query(question, session_id)
        if answer:
            yield answer

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _run_query_workflow(self, init_state: Dict[str, Any]) -> QueryGraphState:  # type: ignore[name-defined]
        """同步执行查询工作流(供 to_thread 调用)."""
        workflow = self._get_query_workflow()
        return workflow.run(init_state)

    def _run_import_workflow(self, init_state: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行导入工作流(供 to_thread 调用)."""
        result = KBImportWorkflow.create_and_run(init_state)  # type: ignore[arg-type]
        return dict(result)

    def _get_query_workflow(self) -> KBQueryWorkflow:
        """懒创建查询工作流单例."""
        if self._query_workflow is None:
            self._query_workflow = KBQueryWorkflow()
        return self._query_workflow
