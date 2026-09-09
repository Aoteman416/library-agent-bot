"""FastAPI 应用入口

启动方式：
    uvicorn app.main:app --reload --port 8000
    或
    python run.py

新增能力（对应需求分析 v2）：
- 关系数据库自动初始化（init_db，DB 不可用则内存演示）
- RAG 索引构建（含 Cross-Encoder 重排序，可选）
- 统一错误码处理（register_exception_handlers）
- 限流中间件（RateLimitMiddleware，RATE_LIMIT_ENABLED=true 生效）
- CORS（允许前端跨域调试）
- 静态前端挂载
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.v1 import admin, auth, book, chat, seat, user
from .config import settings
from .core.errors import register_exception_handlers
from .core.llm_client import llm_client
from .core.logger import logger
from .core.ratelimit import RateLimitMiddleware
from .models.database import db_available, init_db
from .services.rag_service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库与 RAG 索引"""
    logger.info("========== 启动 %s v%s ==========", settings.APP_NAME, settings.APP_VERSION)
    init_db()
    rag_service.build_index()
    logger.info(
        "LLM 模式：%s",
        "真实大模型" if llm_client.enabled else "离线模板（未配置 LLM_API_KEY，可修改 .env 启用）",
    )
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 FastAPI + RAG 的图书馆智能客服",
    lifespan=lifespan,
)

# 统一错误码处理
register_exception_handlers(app)

# CORS（允许前端跨域调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 限流中间件（RATE_LIMIT_ENABLED=true 时生效，否则透传）
app.add_middleware(RateLimitMiddleware)

# 业务路由
API_PREFIX = "/api/v1"
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(book.router, prefix=API_PREFIX)
app.include_router(user.router, prefix=API_PREFIX)
app.include_router(seat.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "db": "enabled" if db_available() else "demo/mock",
        "llm": "enabled" if llm_client.enabled else "offline",
        "embedding": settings.EMBEDDING_MODE,
        "vector_store": settings.VECTOR_STORE_MODE,
        "reranker": "enabled" if settings.RERANKER_ENABLED else "disabled",
    }


# Web 前端（静态资源），挂载在根路径
if settings.WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(settings.WEB_DIR), html=True), name="web")
