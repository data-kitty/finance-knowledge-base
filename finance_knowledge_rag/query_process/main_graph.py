"""
Query main graph — 查询(检索+生成)工作流主图.

LangGraph StateGraph 编排:
  node_intent_confirm
    → (有澄清 answer) → node_answer_output → END
    → (确认主题) → node_search_embedding → node_rrf → node_rerank
      → node_answer_output → END

KBQueryWorkflow 类封装 compile / run / create_and_run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.constants import END
from langgraph.graph import StateGraph

from finance_knowledge_rag.query_process.base import NodeBase
from finance_knowledge_rag.query_process.nodes.node_answer_output import NodeAnswerOutput
from finance_knowledge_rag.query_process.nodes.node_intent_confirm import NodeIntentConfirm
from finance_knowledge_rag.query_process.nodes.node_rerank import NodeRerank
from finance_knowledge_rag.query_process.nodes.node_rrf import NodeRrf
from finance_knowledge_rag.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from finance_knowledge_rag.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)


class KBQueryWorkflow:
    """知识库查询工作流: 意图确认 → 向量检索 → RRF → 重排 → 答案生成."""

    def __init__(self) -> None:
        """初始化工作流: 创建图, 注册节点, 设置路由."""
        self.workflow = StateGraph(QueryGraphState)
        self._init_nodes()
        self._register_nodes()
        self._setup_routes()
        self._compiled_app = None

    def _init_nodes(self) -> None:
        """实例化全部查询节点."""
        self.node_intent_confirm: NodeBase = NodeIntentConfirm()
        self.node_search_embedding: NodeBase = NodeSearchEmbedding()
        self.node_rrf: NodeBase = NodeRrf()
        self.node_rerank: NodeBase = NodeRerank()
        self.node_answer_output: NodeBase = NodeAnswerOutput()

    def _register_nodes(self) -> None:
        """注册节点到工作流."""
        self.workflow.add_node("node_intent_confirm", self.node_intent_confirm)
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

    def _route_after_intent(self, state: QueryGraphState) -> str:
        """意图确认后的条件路由: 有澄清答案直接生成, 否则走检索链路."""
        if state.get("answer"):
            return "node_answer_output"
        return "node_search_embedding"

    def _setup_routes(self) -> None:
        """定义入口与边."""
        self.workflow.set_entry_point("node_intent_confirm")

        # 条件路由: intent_confirm → (answer?) → answer_output | search_embedding
        self.workflow.add_conditional_edges(
            "node_intent_confirm",
            self._route_after_intent,
            {
                "node_answer_output": "node_answer_output",
                "node_search_embedding": "node_search_embedding",
            },
        )

        # 检索链路: search_embedding → rrf → rerank → answer_output → END
        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def compile(self):
        """编译工作流(懒编译, 首次调用时执行)."""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: Dict[str, Any]) -> QueryGraphState:
        """执行查询工作流.

        Args:
            initial_state: 初始图状态(必须包含 session_id / original_query).

        Returns:
            最终状态(含 answer 字段).
        """
        compiled = self.compile()
        if compiled is None:  # pragma: no cover
            raise RuntimeError("Query workflow failed to compile")
        return compiled.invoke(initial_state)

    @classmethod
    def create_and_run(cls, init_state: Dict[str, Any]) -> QueryGraphState:
        """便捷方法: 创建实例并执行."""
        workflow = cls()
        return workflow.run(init_state)
