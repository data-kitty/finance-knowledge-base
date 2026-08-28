"""
MongoDB 聊天历史工具模块。

HistoryMongoTool 提供对 chat_message 集合的保存/查询/清除/更新操作。
所有配置 (mongo_url, mongo_db_name) 均从 rag.config 读取。
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from bson import ObjectId
from pymongo import MongoClient, ASCENDING

from finance_knowledge_rag.config import rag_config

logger = logging.getLogger(__name__)


class HistoryMongoTool:
    """
    管理聊天历史消息的 MongoDB 工具。

    chat_message 集合存储用户/助手消息, 包含 session_id、
    role、text、rewritten_query、item_names、image_urls 和 timestamp 字段。
    """

    def __init__(self):
        """初始化 MongoDB 连接并确保索引存在。"""
        try:
            self.mongo_url = rag_config.mongo.mongo_url
            self.db_name = rag_config.mongo.mongo_db_name

            if not self.mongo_url:
                raise ValueError("MONGO_URL is not configured")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # 创建 (session_id 升序, ts 降序) 索引——幂等操作
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logger.info("MongoDB connected: db=%s", self.db_name)
        except Exception as e:
            logger.exception("MongoDB connection failed: %s", e)
            raise

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: Optional[List[str]] = None,
        image_urls: Optional[List[str]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """
        按 message_id 插入新聊天消息或更新已有消息。

        Args:
            session_id: 会话标识。
            role: 消息角色 ("user" 或 "assistant")。
            text: 消息内容文本。
            rewritten_query: 改写后的查询 (用于用户消息)。
            item_names: 与消息关联的金融主题名称列表。
            image_urls: 图片 URL 列表 (用于带图片的助手消息)。
            message_id: 若提供, 则更新已有消息; 否则插入新消息。

        Returns:
            消息 ID 字符串 (插入或更新的)。
        """
        try:
            document = {
                "session_id": session_id,
                "role": role,
                "text": text,
                "rewritten_query": rewritten_query,
                "item_names": item_names or [],
                "image_urls": image_urls or [],
                "ts": datetime.now().timestamp(),
            }

            if message_id:
                self.chat_message.update_one(
                    {"_id": ObjectId(message_id)},
                    {"$set": document},
                )
                return message_id
            else:
                result = self.chat_message.insert_one(document)
                return str(result.inserted_id)
        except Exception as e:
            logger.exception("save_chat_message failed for session=%s: %s", session_id, e)
            raise

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取会话的最近聊天消息, 按时间戳升序排列。

        Args:
            session_id: 会话标识。
            limit: 返回的最大消息条数。

        Returns:
            消息文档列表。
        """
        try:
            cursor = self.chat_message.find({"session_id": session_id}).sort("ts", ASCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.exception("get_recent_messages failed for session=%s: %s", session_id, e)
            raise

    def clear_history(self, session_id: str) -> int:
        """
        删除会话的全部聊天消息。

        Args:
            session_id: 会话标识。

        Returns:
            删除的消息条数。
        """
        try:
            result = self.chat_message.delete_many({"session_id": session_id})
            return result.deleted_count
        except Exception as e:
            logger.exception("clear_history failed for session=%s: %s", session_id, e)
            raise

    def update_message_item_names(self, message_ids: List[str], item_names: List[str]) -> int:
        """
        按消息 ID 批量更新多条消息的 item_names 字段。

        Args:
            message_ids: 消息 ID 字符串列表。
            item_names: 要设置的金融主题名称列表。

        Returns:
            被修改的文档数量。
        """
        try:
            object_ids = [ObjectId(mid) for mid in message_ids]
            result = self.chat_message.update_many(
                {"_id": {"$in": object_ids}},
                {"$set": {"item_names": item_names}},
            )
            return result.modified_count
        except Exception as e:
            logger.exception("update_message_item_names failed: %s", e)
            raise


# 单例实例
_history_mongo_tool: Optional[HistoryMongoTool] = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取 HistoryMongoTool 单例 (懒初始化)。"""
    global _history_mongo_tool
    if _history_mongo_tool is not None:
        return _history_mongo_tool
    _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


# 模块级便捷函数 (委托给单例)
def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    message_id: Optional[str] = None,
) -> str:
    """委托给单例 HistoryMongoTool.save_chat_message。"""
    return get_history_mongo_tool().save_chat_message(
        session_id=session_id,
        role=role,
        text=text,
        rewritten_query=rewritten_query,
        item_names=item_names,
        image_urls=image_urls,
        message_id=message_id,
    )


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """委托给单例 HistoryMongoTool.get_recent_messages。"""
    return get_history_mongo_tool().get_recent_messages(session_id, limit)


def clear_history(session_id: str) -> int:
    """委托给单例 HistoryMongoTool.clear_history。"""
    return get_history_mongo_tool().clear_history(session_id)


def update_message_item_names(message_ids: List[str], item_names: List[str]) -> int:
    """委托给单例 HistoryMongoTool.update_message_item_names。"""
    return get_history_mongo_tool().update_message_item_names(message_ids, item_names)


class MongoJSONEncoder(json.JSONEncoder):
    """用于 MongoDB ObjectId 和 datetime 的自定义 JSON 编码器。"""

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """使用 MongoDB 安全编码将数据序列化为 JSON。"""
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, cls=MongoJSONEncoder)
