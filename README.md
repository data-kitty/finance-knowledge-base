# Finance Knowledge RAG — 金融知识库 RAG 服务

从金融智能客服后端(finance-customer-service-backend)中拆出的独立 RAG 服务。
原为客服后端 `customer_service/rag` 内嵌模块, 现独立部署, 通过 HTTP 接口
向客服后端等下游提供知识问答与文档导入能力。

## 架构

```
客服后端 (customer-service-backend, :18081)
  └─ knowledge 轨道 → RagHttpClient ──HTTP──▶ 本服务 (finance-knowledge-RAG, :18082)
                                                 ├─ POST /api/rag/query    知识问答
                                                 ├─ POST /api/rag/import   重新导入文档
                                                 ├─ GET  /api/rag/collections  Milvus 状态
                                                 └─ GET  /health          健康检查
```

- 查询链路: 意图确认 → Milvus 混合检索(dense+sparse) → RRF 融合 → DashScope
  reranker 重排 → LLM 答案生成(qwen-flash, 独立于客服主链路 deepseek-v4-flash)
- 导入链路: MD 解析 → 切片 → 主题标签提取 → BGE-M3 向量化 → Milvus 入库
- 技术栈: LangGraph 双工作流 / Milvus 2.x / BGE-M3(FlagEmbedding) / DashScope /
  MongoDB(对话历史) / MinIO(备用)
- 查询失败返回 `answer=""`, 客服后端据此降级到内置 FAQ, 不影响主链路

## 目录结构

```
finance-knowledge-RAG/
├── main.py                       # 入口 (uvicorn)
├── pyproject.toml                # 独立依赖
├── .env / .env.example           # 配置(全部环境变量化, 无硬编码)
├── knowledge_docs/               # 知识文档(6份 .md)
└── finance_knowledge_rag/        # 主包(从 customer_service.rag 迁移, 更名下划线)
    ├── config.py                 # 配置加载(dotenv)
    ├── rag_service.py            # 对外服务: import_documents / query / query_with_state
    ├── api/                      # FastAPI 层(app + rag_router)
    ├── import_process/           # 导入工作流(LangGraph)
    ├── query_process/            # 查询工作流(LangGraph)
    └── utils/                    # embedding / llm / milvus_utils / mongo / reranker
```

## 启动

```bash
# 1. 创建虚拟环境并安装依赖(依赖较大, 含 flagembedding→torch ~2GB)
uv venv .venv
uv pip install -e . --default-index https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置 .env(从 .env.example 复制并填写; 必填项缺失启动即报错)
cp .env.example .env

# 3. 启动(默认 127.0.0.1:18082, 可在 .env 改 APP_PORT)
uv run python main.py
# 或: uv run uvicorn finance_knowledge_rag.api.app:app --port 18082

# 4. 健康检查
curl http://127.0.0.1:18082/health
```

> 提示: 若本机已有客服后端或 01_knowledge-base 的 .venv(已装 pymilvus 2.4.9 /
> flagembedding 等), 可直接复用该解释器运行, 省去 2GB 重装。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/rag/query | 知识问答; body `{question, session_id?}`; 返回 `{answer, session_id, item_names, rewritten_query, error}` |
| POST | /api/rag/import | 重新导入知识文档(幂等); body `{docs_dir?}`; 返回 `{results}` |
| GET  | /api/rag/collections | Milvus 集合与行数(排障) |
| GET  | /health | 健康检查 |

示例:

```bash
curl -X POST http://127.0.0.1:18082/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "个人贷款需要什么条件?", "session_id": "s001"}'

curl -X POST http://127.0.0.1:18082/api/rag/import \
  -H "Content-Type: application/json" -d '{}'
```

## 与客服后端的对接

客服后端通过 `RagHttpClient`(customer_service/knowledge/rag_http_client.py)调用本服务:
- 客服后端 `.env` 配 `RAG_SERVICE_URL=http://127.0.0.1:18082`
- `RAG_SERVICE_URL` 为空或不可达时, 客服后端知识轨道自动降级内置 FAQ
- RAG 专属配置(RAG_LLM_API_KEY / MILVUS_URL / BGE_M3_PATH 等)已迁移到本服务 .env,
  客服后端不再需要

## 排障速查

- 查询返回"未找到"≠ 没路由到 RAG, 而是 RAG 内部某环节失败: 看本服务 uvicorn 日志
  (`Extracted topics: []` = LLM 提取失败; 401/429 = key 无效/冷却)
- Milvus 集合检查: `GET /api/rag/collections`
- 主题名是文件名 fallback(如 "01_个人贷款业务指南"): RAG_LLM_API_KEY 无效所致,
  换有效 key 后重跑 `POST /api/rag/import` 即恢复语义化主题(幂等)

## 依赖版本坑(踩过)

- `pymilvus` 必须 `==2.4.9`(3.x 无 `pymilvus.model`, BGE-M3 嵌入函数无法导入)
- `setuptools<81`、`marshmallow<4`(新版本移除 pkg_resources / __version_info__)
- LangGraph 会过滤 TypedDict schema 外的键: `ImportGraphState` 必须声明
  `local_file_path` 字段, 否则入口节点取不到文件路径
