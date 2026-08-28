"""
RAG 知识库服务入口。

启动: uvicorn main:app 或 python main.py
监听: 读取 .env 的 APP_HOST / APP_PORT(默认 127.0.0.1:18082)
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# 先加载 .env(确保 APP_HOST / APP_PORT 可用), 再导入应用
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def main() -> None:
    """启动 RAG 服务。"""
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "18082"))
    uvicorn.run(
        "finance_knowledge_rag.api.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
