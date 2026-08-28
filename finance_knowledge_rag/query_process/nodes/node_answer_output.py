"""
查询工作流的答案生成节点。

从参考内容 + 聊天历史 + 金融主题 + 用户问题组装提示词,
然后调用 LLM 生成 (流式) 答案。
将答案保存到 MongoDB 聊天历史。
"""
import logging
from typing import Any, Dict, List, Tuple

from finance_knowledge_rag.query_process.base import NodeBase
from finance_knowledge_rag.query_process.prompt import ANSWER_PROMPT
from finance_knowledge_rag.query_process.state import QueryGraphState
from finance_knowledge_rag.utils.llm import get_llm_client
from finance_knowledge_rag.utils.mongo import save_chat_message

logger = logging.getLogger(__name__)


class NodeAnswerOutput(NodeBase):
    """
    答案生成节点。

    若 state 已有 'answer' (来自意图澄清), 直接返回。
    否则组装提示词并调用 LLM 生成答案,
    可通过 is_stream 标志选择流式输出。
    """

    name = "node_answer_output"

    # 上下文最大字符数 (参考文档 + 历史)
    MAX_CONTEXT_CHARS = 12000

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        生成最终答案。

        Args:
            state: 可能包含 'answer' (澄清) 或 'reranked_docs'。

        Returns:
            包含 'answer' 的字典, 可选包含 'prompt'。
        """
        # 1. 检查答案是否已存在 (来自意图确认)
        answer = state.get("answer", "")

        if answer:
            # 澄清答案——无需调用 LLM
            return {"answer": answer}

        # 2. 从重排文档 + 历史 + 主题 + 问题组装提示词
        prompt = self._step1_construct_prompt(state)

        # 3. 通过 LLM 生成答案
        is_stream = state.get("is_stream", False)
        answer = self._step2_generate_response(prompt, is_stream)

        # 4. 保存到 MongoDB 历史
        self._step3_write_history(state, answer)

        return {"answer": answer, "prompt": prompt}

    def _step1_construct_prompt(self, state: QueryGraphState) -> str:
        """
        从以下内容组装 LLM 提示词:
        - reranked_docs (参考内容)
        - history (聊天历史)
        - item_names (金融主题)
        - rewritten_query (问题)
        """
        char_budget = self.MAX_CONTEXT_CHARS

        # 1. 获取问题和主题名称
        question = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])

        # 2. 格式化参考文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs", []), char_budget
        )

        # 3. 格式化聊天历史
        history_str, char_budget = self._format_chat_history(
            state.get("history", []), char_budget
        )

        # 4. 格式化 item_names
        item_names_str = ", ".join(item_names) if item_names else "无指定主题"

        # 5. 组装提示词
        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        return prompt

    def _format_reranked_docs(
        self, reranked_docs: List[Dict], char_budget: int
    ) -> Tuple[str, int]:
        """将重排文档格式化为带元数据标签的参考文本。"""
        formatted_lines: List[str] = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content", "")

            # 构建元数据标签
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value and field_value != "None":
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(
        self, chat_history: List[Dict], char_budget: int
    ) -> Tuple[str, int]:
        """为提示词格式化聊天历史。"""
        formatted_lines: List[str] = []
        used_chars = 0

        role_labels = {"user": "用户", "assistant": "助手"}

        for msg in chat_history:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if not text or role not in role_labels:
                continue

            line = f"{role_labels[role]}: {text}"
            used_chars += len(line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _step2_generate_response(self, prompt: str, is_stream: bool) -> str:
        """
        通过 LLM 生成答案。

        若 is_stream 为 True, 产出增量 token (收集后作为
        单个字符串返回; 调用方可适配 SSE 流式输出)。
        """
        llm = get_llm_client()

        if is_stream:
            # 流式: 收集所有分块
            final_text = ""
            try:
                for chunk in llm.stream(prompt):
                    delta = chunk.content
                    if delta:
                        final_text += delta
            except Exception as e:
                logger.exception("Streaming generation failed: %s", e)
                raise
            return final_text
        else:
            # 非流式: 单次调用
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                logger.exception("LLM invocation failed: %s", e)
                raise

    def _step3_write_history(self, state: QueryGraphState, answer: str) -> None:
        """将助手的答案保存到 MongoDB 聊天历史。"""
        session_id = state.get("session_id", "")
        item_names = state.get("item_names", [])

        try:
            if answer and session_id:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                )
        except Exception as e:
            # 历史写入失败不应影响主流程
            logger.error("Failed to write answer to history: %s", e)
