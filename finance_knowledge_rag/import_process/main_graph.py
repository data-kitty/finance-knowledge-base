"""
导入主图 — 定义完整的导入 (文档入库) 工作流。

LangGraph StateGraph 编排:
  entry -> md_parse -> document_split -> item_extract -> embedding -> import_milvus -> END

KBImportWorkflow 类封装 compile、run 和 create_and_run。
"""
import logging
from typing import Optional

from langgraph.constants import END
from langgraph.graph import StateGraph

from finance_knowledge_rag.import_process.nodes.node_entry import NodeEntry
from finance_knowledge_rag.import_process.nodes.node_md_parse import NodeMdParse
from finance_knowledge_rag.import_process.nodes.node_document_split import NodeDocumentSplit
from finance_knowledge_rag.import_process.nodes.node_item_extract import NodeItemExtract
from finance_knowledge_rag.import_process.nodes.node_embedding import NodeEmbedding
from finance_knowledge_rag.import_process.nodes.node_import_milvus import NodeImportMilvus
from finance_knowledge_rag.import_process.state import ImportGraphState

logger = logging.getLogger(__name__)


class KBImportWorkflow:
    """
    知识库导入工作流。

    封装 LangGraph 工作流的构建、编译与执行。
    支持多个独立实例。
    """

    def __init__(self):
        """初始化工作流: 创建状态图, 注册节点, 设置路由。"""
        self.workflow = StateGraph(ImportGraphState)
        self._init_nodes()
        self._register_nodes()
        self._setup_routes()
        self._compiled_app = None

    def _init_nodes(self):
        """实例化全部业务节点。"""
        self.node_entry = NodeEntry()
        self.node_md_parse = NodeMdParse()
        self.node_document_split = NodeDocumentSplit()
        self.node_item_extract = NodeItemExtract()
        self.node_embedding = NodeEmbedding()
        self.node_import_milvus = NodeImportMilvus()

    def _register_nodes(self):
        """将全部节点注册到工作流。"""
        self.workflow.add_node("node_entry", self.node_entry)
        self.workflow.add_node("node_md_parse", self.node_md_parse)
        self.workflow.add_node("node_document_split", self.node_document_split)
        self.workflow.add_node("node_item_extract", self.node_item_extract)
        self.workflow.add_node("node_embedding", self.node_embedding)
        self.workflow.add_node("node_import_milvus", self.node_import_milvus)

    def _setup_routes(self):
        """定义入口点与边。"""
        # 入口点
        self.workflow.set_entry_point("node_entry")

        # 线性边: entry -> md_parse -> split -> item_extract -> embedding -> import_milvus -> END
        self.workflow.add_edge("node_entry", "node_md_parse")
        self.workflow.add_edge("node_md_parse", "node_document_split")
        self.workflow.add_edge("node_document_split", "node_item_extract")
        self.workflow.add_edge("node_item_extract", "node_embedding")
        self.workflow.add_edge("node_embedding", "node_import_milvus")
        self.workflow.add_edge("node_import_milvus", END)

    def compile(self):
        """编译工作流 (懒编译, 仅在首次调用时执行)。"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: ImportGraphState) -> ImportGraphState:
        """
        执行导入工作流。

        Args:
            initial_state: 初始图状态 (必须包含 'local_file_path')。

        Returns:
            所有节点执行完成后的最终状态。
        """
        if not self._compiled_app:
            self.compile()
        return self._compiled_app.invoke(initial_state)

    @classmethod
    def create_and_run(cls, init_state: ImportGraphState) -> ImportGraphState:
        """便捷方法: 创建工作流实例并执行。"""
        workflow = cls()
        return workflow.run(init_state)


if __name__ == "__main__":
    import sys
    import os

    # 快速测试: 在单个 .md 文件上运行
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默认: knowledge_docs 中第一个 .md 文件
        docs_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "knowledge_docs"
        )
        docs_dir = os.path.abspath(docs_dir)
        md_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
        if not md_files:
            print("No .md files found in knowledge_docs")
            sys.exit(1)
        file_path = os.path.join(docs_dir, md_files[0])

    init_state = {
        "task_id": "import_test",
        "local_file_path": file_path,
    }

    result = KBImportWorkflow.create_and_run(init_state)
    logger.info("Import complete. Final state keys: %s", list(result.keys()))
