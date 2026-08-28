# 金融知识库 RAG 服务镜像
# 构建: docker build -t finance-rag .
# 运行环境变量见 docker-compose.yaml rag 服务
# BGE-M3 模型不打包进镜像: 通过 volume 挂载 /models/bge-m3(见 download_bge_m3.sh)

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先复制依赖清单, 利用构建缓存
COPY pyproject.toml uv.lock ./
# 依赖含 torch/flagembedding(~2GB), 走清华镜像加速
RUN uv sync --frozen --no-dev --default-index https://pypi.tuna.tsinghua.edu.cn/simple

# 复制源码(.env 已被 .dockerignore 排除)
COPY . .

EXPOSE 18082

CMD ["uv", "run", "python", "main.py"]
