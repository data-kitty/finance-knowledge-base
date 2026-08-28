"""
导入工作流的入口节点。

校验输入文件类型 (.md) 并提取 file_title (不带扩展名的文件名)。
本项目仅支持 Markdown 文件。
"""
import os
from os.path import splitext
from typing import Any, Dict

from finance_knowledge_rag.import_process.base import NodeBase
from finance_knowledge_rag.import_process.state import ImportGraphState


class NodeEntry(NodeBase):
    """
    入口节点: 任务分发。

    检查输入文件类型并设置相应的流程控制标志。
    本项目仅支持 .md 文件。
    """

    name = "node_entry"

    def process(self, state: ImportGraphState) -> Dict[str, Any]:
        """
        校验文件路径、检查文件类型、提取 file_title。

        Args:
            state: 必须包含 'local_file_path'。

        Returns:
            包含 is_md_read_enabled、md_path、file_title 的字典。

        Raises:
            ValueError: 当文件路径为空或文件类型不受支持时。
        """
        local_file_path = state.get("local_file_path", "")
        if not local_file_path:
            raise ValueError("local_file_path must be specified")

        # 提取文件标题 (不带扩展名的文件名)
        file_title = splitext(os.path.basename(local_file_path))[0]

        # 检查文件类型——本项目仅支持 .md 文件
        if local_file_path.lower().endswith(".md"):
            return {
                "is_md_read_enabled": True,
                "md_path": local_file_path,
                "file_title": file_title,
            }
        else:
            # 提取扩展名用于错误提示
            dot_idx = local_file_path.rfind(".")
            ext = local_file_path[dot_idx + 1:] if dot_idx != -1 else "unknown"
            raise ValueError(f"Unsupported file type: .{ext}. Only .md files are supported.")
