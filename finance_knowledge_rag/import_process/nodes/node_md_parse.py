"""
导入工作流的 Markdown 解析节点。

直接读取 .md 文件内容 — 本项目所有知识文档均为 Markdown,
无需 PDF 转换。
"""
import logging
from typing import Any, Dict

from finance_knowledge_rag.import_process.base import NodeBase
from finance_knowledge_rag.import_process.state import ImportGraphState

logger = logging.getLogger(__name__)


class NodeMdParse(NodeBase):
    """
    Markdown 解析节点: 读取 .md 文件并将 md_content 写入状态。

    简单实现: 打开文件, 读取内容, 返回。
    本项目无需图片提取或 PDF 转换。
    """

    name = "node_md_parse"

    def process(self, state: ImportGraphState) -> Dict[str, Any]:
        """
        从 md_path 读取 markdown 文件内容。

        Args:
            state: 必须包含 'md_path'。

        Returns:
            包含 md_content 的字典。

        Raises:
            ValueError: 当 md_path 为空或文件无法读取时。
        """
        md_path = state.get("md_path", "")
        if not md_path:
            raise ValueError("md_path must be specified")

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            if not md_content.strip():
                raise ValueError(f"Markdown file is empty: {md_path}")

            logger.info("Read markdown file: %s (%d chars)", md_path, len(md_content))
            return {"md_content": md_content}

        except FileNotFoundError:
            raise ValueError(f"Markdown file not found: {md_path}")
        except Exception as e:
            raise ValueError(f"Failed to read markdown file '{md_path}': {e}")
