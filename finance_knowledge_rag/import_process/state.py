"""
导入流程图状态定义。

定义 ImportGraphState TypedDict — 在导入 (文档入库) 工作流
所有节点间流转的数据结构。
"""
from typing import TypedDict, List, Dict, Any


class ImportGraphState(TypedDict):
    """
    导入工作流的图状态。

    包含导入节点产生和消费的所有数据字段:
    任务跟踪、流程控制标志、文件路径、内容与向量。
    """
    # 任务追踪
    task_id: str  # 唯一任务 ID, 用于日志/追踪

    # 输入文件路径(由入口节点读取, 必须由调用方传入)
    local_file_path: str  # 待导入的 Markdown 文件路径

    # 流程控制开关
    is_md_read_enabled: bool  # 是否启用 MD 读取路径

    # 文件元数据
    file_title: str  # 文件标题 (不带扩展名的文件名)
    md_path: str  # Markdown 文件路径

    # 内容数据
    md_content: str  # 完整 markdown 内容
    chunks: List[Dict[str, Any]]  # 切片字典列表
    item_name: str  # 识别出的金融主题名称 (例如 "个人贷款")

    # 向量数据
    embeddings_content: List[Dict[str, Any]]  # 带向量的切片, 准备写入 Milvus
