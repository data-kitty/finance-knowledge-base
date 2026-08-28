"""
导入工作流的文档切片节点。

使用多步策略将 markdown 内容切分为块:
1. 按 MD 标题切分 (初步分节)
2. 切分过长的节 (最大 500 字符)
3. 合并过短的块 (最小 100 字符)
4. 50 字符的重叠窗口以保持上下文连续

输出: 切片字典列表 [{title, content, file_title, parent_title, part}]。
"""
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_knowledge_rag.import_process.base import NodeBase
from finance_knowledge_rag.import_process.state import ImportGraphState

logger = logging.getLogger(__name__)


class NodeDocumentSplit(NodeBase):
    """
    文档切片节点: 智能 markdown 分块。

    按标题将 markdown 内容切分为语义连贯的块,
    再进一步优化 (切长、并短) 以获得最佳检索质量。
    """

    name = "node_document_split"

    # 切片内容最大长度——触发二次切分
    DEFAULT_MAX_CONTENT_LENGTH = 500

    # 切片内容最小长度——触发与相邻块合并
    MIN_CONTENT_LENGTH = 100

    # 二次切分的重叠窗口大小
    DEFAULT_WINDOW_OVERLAP = 50

    def process(self, state: ImportGraphState) -> Dict[str, Any]:
        """
        执行多步分块流水线。

        Args:
            state: 必须包含 'md_content' 和 'file_title'。

        Returns:
            包含 'chunks' 的字典 — 切片字典列表。
        """
        # 步骤 1: 校验输入
        content, file_title = self._step1_get_inputs(state)

        # 步骤 2: 按 MD 标题切分 (初步分节)
        sections, title_count, lines_count = self._step2_split_by_titles(content, file_title)

        # 步骤 3: 优化切片 (切长并短, 补充元数据)
        sections = self._step4_refine_chunks(sections)

        # 步骤 4: 记录统计信息
        self._step5_print_stats(lines_count, sections)

        return {"chunks": sections}

    def _step1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """校验并从 state 中提取 md_content 和 file_title。"""
        file_title = state.get("file_title")
        if not file_title:
            raise ValueError("file_title must not be empty")

        md_content = state.get("md_content")
        if not md_content:
            raise ValueError("md_content must not be empty")

        return md_content, file_title

    def _step2_split_by_titles(
        self, content: str, file_title: str
    ) -> Tuple[List[Dict[str, str]], int, int]:
        """
        按标题 (1-6 级) 切分 markdown。

        处理代码围栏 (``` 块), 避免误判标题。
        """
        # 匹配 MD 标题: 1-6 个 '#' 后跟空白和文本
        title_pattern = r"^\s*#{1,6}\s+.+"

        in_code_block = False
        code_fence = None
        current_lines: List[str] = []
        sections: List[Dict[str, str]] = []
        current_title = ""
        title_count = 0

        def _flush_section():
            nonlocal title_count
            if not current_lines:
                return
            title_count += 1
            sections.append({
                "file_title": file_title,
                "title": current_title or "无标题",
                "content": "\n".join(current_lines),
            })

        # 统一换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        lines = content.split("\n")

        for line in lines:
            stripped = line.strip()

            # 检测代码围栏 (``` 或 ~~~)
            code_match = re.match(r"^(`{3,}|~{3,})", stripped)
            if code_match:
                marker = code_match.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_fence = marker
                elif code_fence == marker:
                    in_code_block = False
                    code_fence = None

            # 检测标题 (仅在代码块外)
            is_title = not in_code_block and re.match(title_pattern, stripped)
            if is_title:
                _flush_section()
                current_title = stripped
                current_lines = [current_title]
            else:
                current_lines.append(stripped)

        # 刷出最后一个节
        _flush_section()

        logger.info(
            "Initial split: %d headings, %d sections, %d lines",
            title_count, len(sections), len(lines),
        )
        return sections, title_count, len(lines)

    def _step4_refine_chunks(
        self, sections: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """优化切片: 切分过长节、合并过短块、补充元数据。"""
        # 1. 切分过长的节
        refined_split: List[Dict[str, str]] = []
        for sec in sections:
            refined_split.extend(self._split_long_section(sec))
        logger.info("After splitting long sections: %d chunks", len(refined_split))

        # 2. 合并过短的块
        final_sections = self._merge_short_sections(refined_split)
        logger.info("After merging short sections: %d chunks", len(final_sections))

        # 3. 为缺少 parent_title 的节补充 parent_title 与 part 元数据
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title", "")
                sec["part"] = 0

        logger.info("Final refined chunks: %d", len(final_sections))
        return final_sections

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """切分内容超过最大长度的节。"""
        content = section.get("content", "")

        if len(content) <= self.DEFAULT_MAX_CONTENT_LENGTH:
            return [section]

        # 不切分 HTML 表格——保留结构
        if "<table" in content.lower():
            return [section]

        title = section.get("title", "")
        prefix = f"{title}\n\n"
        available_len = self.DEFAULT_MAX_CONTENT_LENGTH - len(prefix)

        if available_len <= 0:
            logger.warning("Title too long, cannot split: %s", title)
            return [section]

        logger.info("Splitting long section: %s", title)

        # 从内容中移除标题前缀
        body = content
        if title and body.startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=self.DEFAULT_WINDOW_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        sub_sections: List[Dict[str, str]] = []
        for idx, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue

            full_text = prefix + text
            sub_sections.append({
                "title": f"{title} - {idx}",
                "content": full_text,
                "parent_title": title,
                "part": idx,
                "file_title": section.get("file_title", ""),
            })

        return sub_sections

    def _merge_short_sections(
        self, sections: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """合并具有相同 parent_title 的相邻短块。"""
        if not sections:
            return []

        merged: List[Dict[str, str]] = []
        current_chunk: Dict[str, str] | None = None

        for sec in sections:
            if current_chunk is None:
                current_chunk = sec
                continue

            is_current_short = len(current_chunk.get("content", "")) < self.MIN_CONTENT_LENGTH
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                parent_title = sec.get("parent_title", "")
                next_content = sec.get("content", "")

                # 移除下一个块中重复的标题前缀
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()

                current_chunk["content"] += "\n\n" + next_content
                if "part" in sec:
                    current_chunk["part"] = sec["part"]

                logger.info(
                    "Merged short chunk: %s -> %d chars",
                    current_chunk.get("parent_title"),
                    len(current_chunk["content"]),
                )
            else:
                merged.append(current_chunk)
                current_chunk = sec

        if current_chunk is not None:
            merged.append(current_chunk)

        return merged

    def _step5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """记录分块统计信息。"""
        logger.info("-" * 50 + " document split stats " + "-" * 50)
        logger.info("MD total lines: %d", lines_count)
        logger.info("Final chunk count: %d", len(sections))
