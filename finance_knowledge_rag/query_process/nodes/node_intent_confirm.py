"""
查询工作流的意图确认节点。

从用户问题中提取金融主题名称, 改写查询, 对 item_name_collection
执行向量检索以对齐主题, 并按置信度打分
(>0.85 确认, >=0.6 候选, <0.6 拒绝)。

返回已确认主题的 item_names, 或请求澄清的答案。
"""
import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from finance_knowledge_rag.config import rag_config
from finance_knowledge_rag.query_process.base import NodeBase
from finance_knowledge_rag.query_process.prompt import (
    ITEM_NAME_EXTRACT_TEMPLATE,
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT,
)
from finance_knowledge_rag.query_process.state import QueryGraphState
from finance_knowledge_rag.utils.embedding import generate_embeddings
from finance_knowledge_rag.utils.llm import get_llm_client
from finance_knowledge_rag.utils.milvus_utils import create_hybrid_search_request, hybrid_search
from finance_knowledge_rag.utils.mongo import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)

logger = logging.getLogger(__name__)


class NodeIntentConfirm(NodeBase):
    """
    意图确认节点。

    使用 LLM 从用户问题中提取金融主题名称,
    然后通过混合检索打分与 Milvus item_name_collection 对齐。
    """

    name = "node_intent_confirm"

    # 置信度阈值
    CONFIRM_THRESHOLD = 0.85  # >0.85 -> 直接确认
    CANDIDATE_THRESHOLD = 0.6  # >=0.6 -> 候选 (询问澄清)

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        执行意图确认流水线。

        流程:
        1. 校验参数 (session_id, original_query)
        2. 从 MongoDB 获取聊天历史
        3. 保存用户的当前问题
        4. 提取金融主题 + 改写查询 (LLM)
        5. 向量化并混合检索 item_name_collection
        6. 按分数对齐主题
        7. 检查确认状态 (确认/候选/拒绝)
        8. 写入历史
        9. 返回 item_names 或澄清答案
        """
        # 步骤 1: 校验参数
        session_id, original_query = self._step1_validate_param(state)

        # 步骤 2: 获取聊天历史
        history = get_recent_messages(session_id)

        # 步骤 3: 保存用户的当前问题
        message_id = save_chat_message(session_id, "user", original_query)

        # 步骤 4: 提取金融主题 + 改写查询
        extract_result = self._step4_extract_info(original_query, history)
        item_names = extract_result.get("item_names", [])
        rewritten_query = extract_result.get("rewritten_query", original_query)
        logger.info("Extracted topics: %s, rewritten: '%s'", item_names, rewritten_query)

        # 步骤 5-6: 向量化并对齐 (仅在提取到主题时)
        align_result: Dict = {}
        if item_names:
            query_results = self._step5_vectorize_and_query(item_names)
            align_result = self._step6_align_item_names(query_results)
        else:
            logger.warning("No topics extracted, skipping vector alignment")

        # 步骤 7: 检查确认状态
        dict_result = self._step7_check_confirmation(align_result, history)

        # 步骤 8: 写入历史
        self._step8_write_history(
            dict_result, session_id, original_query, rewritten_query, message_id
        )

        # 步骤 9: 返回结果
        return {
            "history": get_recent_messages(session_id),
            "rewritten_query": rewritten_query,
            "item_names": dict_result.get("item_names", []),
            "answer": dict_result.get("answer", ""),
        }

    def _step1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:
        """校验 session_id 和 original_query。"""
        session_id = state.get("session_id", "")
        if not session_id:
            raise ValueError("session_id is required")

        original_query = state.get("original_query", "")
        if not original_query:
            raise ValueError("original_query is required")

        return session_id, original_query

    def _step4_extract_info(self, original_query: str, history: list) -> Dict:
        """使用 LLM 提取金融主题名称并改写查询。"""
        try:
            # 构建历史文本 (最近的在最前)
            history_text = ""
            for msg in reversed(history):
                history_text += f"{msg.get('role', '')}: {msg.get('text', '')}\n"

            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                original_query=original_query,
            )

            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            # 使用 item_model 并开启 JSON 模式
            llm = get_llm_client(model=rag_config.llm.item_model, json_mode=True)
            response = llm.invoke(messages)

            content = response.content.strip()

            # 清理 markdown 代码块包裹
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")

            result = json.loads(content)

            # 确保必需键存在
            if "item_names" not in result:
                result["item_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = original_query

            # 清理 item_names 中的空白字符
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

            return result

        except Exception as e:
            logger.exception("Topic extraction failed: %s", e)
            return {"item_names": [], "rewritten_query": original_query}

    def _step5_vectorize_and_query(self, item_names: List[str]) -> List[Dict]:
        """
        对每个主题名称向量化, 并对 item_name_collection
        执行混合检索。

        返回 {extracted_name, matches: [{item_name, score}]} 列表。
        """
        embeddings = generate_embeddings(item_names)
        dense_vectors = embeddings["dense"]
        sparse_vectors = embeddings["sparse"]

        results: List[Dict] = []
        collection_name = rag_config.milvus.item_name_collection

        for i, item_name in enumerate(item_names):
            reqs = create_hybrid_search_request(
                dense_vector=dense_vectors[i],
                sparse_vector=sparse_vectors[i],
            )

            search_result = hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                output_fields=["item_name"],
            )

            hits = search_result[0] if search_result else []
            matches = [
                {
                    "item_name": hit.get("entity", {}).get("item_name"),
                    "score": hit.get("distance"),
                }
                for hit in hits
            ]

            results.append({
                "extracted_name": item_name,
                "matches": matches,
            })

        return results

    def _step6_align_item_names(self, query_results: List[Dict]) -> Dict:
        """
        按 Milvus 检索分数对齐提取的主题。

        规则 (优先级 a > b > c):
        a. 分数 > 0.85 -> 直接确认该主题
        b. 无 > 0.85 但分数 >= 0.6 -> 取前 3 个作为候选
        c. 无 >= 0.6 -> 拒绝 (不返回主题)
        """
        confirmed_item_names: List[str] = []
        options: List[str] = []

        for res in query_results:
            matches = res.get("matches", [])
            if not matches:
                continue

            # 分数高于 0.85 -> 确认
            high = [m for m in matches if m.get("score", 0) > self.CONFIRM_THRESHOLD]
            # 分数 >= 0.6 -> 候选
            mid = [m for m in matches if m.get("score", 0) >= self.CANDIDATE_THRESHOLD]

            if high:
                confirmed_item_names += [m.get("item_name") for m in high]
                continue

            if mid:
                options += [m.get("item_name") for m in mid[:3]]

            # 无 >= 0.6 -> 拒绝
            logger.info("No score >= %.1f for '%s', rejected", self.CANDIDATE_THRESHOLD, res.get("extracted_name"))

        return {
            "confirmed_item_names": list(set(confirmed_item_names)),
            "options": list(set(options)),
        }

    def _step7_check_confirmation(self, align_result: Dict, history: list) -> Dict:
        """
        确定确认结果。

        分支 A (已确认): 设置 item_names, 批量更新历史。
        分支 B (候选): 生成澄清答案。
        分支 C (无结果): 生成拒绝答案。
        """
        confirmed_item_names = align_result.get("confirmed_item_names", [])
        options = align_result.get("options", [])

        # 分支 A: 已确认主题
        if confirmed_item_names:
            # 收集没有 item_names 的历史消息 ID
            ids_to_update = [
                str(msg.get("_id"))
                for msg in history
                if not msg.get("item_names")
            ]
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed_item_names)

            return {"item_names": confirmed_item_names, "answer": ""}

        # 分支 B: 候选——询问用户澄清
        if options:
            option_str = "、".join(options)
            return {
                "item_names": [],
                "answer": f"您是想咨询以下哪个主题：{option_str}？请明确一下。",
            }

        # 分支 C: 无结果
        return {
            "item_names": [],
            "answer": "抱歉，未找到相关的金融业务信息，请提供更准确的问题描述。",
        }

    def _step8_write_history(
        self,
        dict_result: Dict,
        session_id: str,
        original_query: str,
        rewritten_query: str,
        message_id: str,
    ) -> None:
        """若存在澄清/拒绝答案, 则写入历史。"""
        if dict_result.get("answer"):
            # 插入助手澄清消息
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=dict_result.get("answer"),
                rewritten_query="",
                item_names=[],
            )

            # 用 rewritten_query 和 item_names 更新用户的原始消息
            save_chat_message(
                session_id=session_id,
                role="user",
                text=original_query,
                rewritten_query=rewritten_query,
                item_names=dict_result.get("item_names", []),
                message_id=message_id,
            )
