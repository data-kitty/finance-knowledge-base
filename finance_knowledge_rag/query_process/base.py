"""
查询流程节点基类。

定义 NodeBase 抽象基类, 其标准化的 __call__ 会记录执行日志,
并提供抽象 process() 方法供子类实现。
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from finance_knowledge_rag.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)


class NodeBase(ABC):
    """
    所有查询工作流节点的抽象基类。

    子类必须:
    1. 设置 `name` 类属性为唯一的节点标识。
    2. 实现包含节点特定逻辑的 `process()` 方法。
    """

    name: str = "node_base"

    def __init__(self):
        """强制子类设置唯一的 name。"""
        if self.name == "node_base":
            raise ValueError(f"{self.__class__.__name__} must set a unique 'name' attribute")

    def __call__(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        节点执行入口 — 用日志和错误处理包装 process()。

        Args:
            state: 当前图状态。

        Returns:
            process() 返回的更新后状态字段。

        Raises:
            Exception: 重新抛出 process() 中的任何异常。
        """
        try:
            logger.info("[%s] starting...", self.name)
            result = self.process(state)
            logger.info("[%s] finished", self.name)
            return result
        except Exception as e:
            logger.exception("[%s] execution failed: %s", self.name, e)
            raise

    @abstractmethod
    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        """
        核心处理逻辑 — 必须由子类实现。

        Args:
            state: 当前图状态。

        Returns:
            需要更新的状态字段字典。
        """
        pass
