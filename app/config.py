"""应用配置管理"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
# 加载 .env 配置（若存在）
load_dotenv(BASE_DIR / ".env")

# 国内免费/低价 LLM 预设（选一个即可，填入 LLM_PROVIDER 自动套用）
# 所有 Provider 都兼容 OpenAI /chat/completions 接口
LLM_PROVIDERS = {
    # SiliconFlow：聚合多家开源模型，个人开发者最常用，有免费额度
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-chat",  # 或 Qwen/Qwen2.5-7B-Instruct
        "note": "https://siliconflow.cn 注册后在「API 密钥」页获取",
    },
    # 阿里云 DashScope：百炼平台，qwen-turbo 新用户免费
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",  # 或 qwen-plus / qwen-max
        "note": "https://bailian.console.aliyun.com 创建 API-KEY",
    },
    # 百度千帆：ERNIE 系列有免费额度
    "qianfan": {
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ERNIE-Speed-8K",
        "note": "https://qianfan.cloud.baidu.com 领取 API Key",
    },
    # 豆包（字节方舟）：有免费额度
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-pro-32k",
        "note": "https://console.volcengine.com/ark 创建 API Key 与推理接入点",
    },
    # One-API 自建聚合（默认端口 3000）
    "oneapi": {
        "base_url": "http://127.0.0.1:3000/v1",
        "model": "gpt-3.5-turbo",
        "note": "部署 One-API 后本地聚合任意模型",
    },
    # 标准 OpenAI
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "note": "",
    },
}


class Settings:
    """全局配置（从环境变量 / .env 读取）"""

    # ---- 服务 ----
    APP_NAME: str = os.getenv("APP_NAME", "图书馆智能客服")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ---- LLM ----
    # 方案一：LLM_PROVIDER（siliconflow/dashscope/qianfan/doubao/oneapi/openai）
    #          + LLM_API_KEY 由 Provider 自动填 base_url / model
    # 方案二：直接手动设置 LLM_BASE_URL + LLM_MODEL + LLM_API_KEY
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "").strip().lower()
    # 根据 provider 自动套用默认值（环境变量显式设置的优先）
    _provider_cfg = LLM_PROVIDERS.get(LLM_PROVIDER, {}) if LLM_PROVIDER else {}

    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "") or _provider_cfg.get("model", "") or "gpt-4o-mini"
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "") or _provider_cfg.get("base_url", "") or "https://api.openai.com/v1"
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "30"))

    # 闲聊模式：unknown 意图是否走 LLM 做基础对话
    CHAT_FALLBACK_TO_LLM: bool = os.getenv("CHAT_FALLBACK_TO_LLM", "true").lower() == "true"
    # 系统提示词：用在闲聊通道，可覆盖
    CHAT_SYSTEM_PROMPT: str = os.getenv(
        "CHAT_SYSTEM_PROMPT",
        (
            "你是图书馆智能客服小助手。请：\n"
            "1. 用亲切、简洁的中文回答用户的闲聊或问候\n"
            "2. 自称「图书馆智能客服」，主动引导用户问：查书、预约座位、续借、开放时间等\n"
            "3. 不回答与图书馆无关的政治、色情、暴力内容，遇到此类问题请礼貌说明只能回答图书馆相关问题\n"
            "4. 不编造馆藏数据或规章制度，涉及具体数据请建议用户通过图书馆官网或总服务台核实"
        ),
    )

    # ---- RAG ----
    # Embedding：ngram(零依赖稀疏 TF-IDF) | external(稠密 BGE / sentence-transformers)
    EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "ngram")
    # 稠密模型配置：资源有限时可换 BAAI/bge-small-zh-v1.5（约 100MB）
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    # bge 系列官方建议：查询侧加指令前缀，与文档侧保持同一语义空间
    EMBEDDING_QUERY_PREFIX: str = os.getenv(
        "EMBEDDING_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章："
    )

    # 向量库：memory(进程内稀疏索引) | chroma(持久化稠密索引)
    VECTOR_STORE_MODE: str = os.getenv("VECTOR_STORE_MODE", "memory")
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma")
    )
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "library_knowledge")

    TOP_K: int = int(os.getenv("TOP_K", "10"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ---- 重排序（Cross-Encoder 精排，对应 v2 3.1 Step3 / 3.2.4）----
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

    # ---- 自适应分块策略（对应 v2 3.2.1：按文档类型选择分块参数）----
    CHUNK_STRATEGY: dict = {
        # 政策文件：大块，保留上下文
        "policy_doc": {"chunk_size": 500, "chunk_overlap": 100},
        # FAQ：按问答对分块，无需重叠
        "faq_doc": {"chunk_size": 300, "chunk_overlap": 0},
        # 使用指南：按章节分块，中等重叠
        "guide_doc": {"chunk_size": 800, "chunk_overlap": 150},
    }

    # ---- 缓存（语义缓存支持 Redis 后端，对应 v2 9.1 缓存）----
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "") or ""
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "3600"))
    REDIS_MAX_SIZE: int = int(os.getenv("REDIS_MAX_SIZE", "500"))

    # ---- 关系数据库（MySQL / 兼容 SQLite 内存演示，对应 v2 5.1）----
    DB_ENABLED: bool = os.getenv("DB_ENABLED", "false").lower() == "true"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:password@localhost:3306/library_service",
    )
    # DB_ENABLED=false 时是否用内存 SQLite 演示（自动建表 + 灌 Demo 数据）
    DB_DEMO_MODE: bool = os.getenv("DB_DEMO_MODE", "true").lower() == "true"

    # ---- 鉴权（JWT + Refresh Token，对应 v2 9.1 认证）----
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    # 管理员演示凭证（真实场景来自 users 表 role=admin）
    ADMIN_USER_ID: str = os.getenv("ADMIN_USER_ID", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

    # ---- 限流（对应 v2 9.1 rate_limiting）----
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_PER_USER: int = int(os.getenv("RATE_LIMIT_PER_USER", "30"))
    RATE_LIMIT_PER_IP: int = int(os.getenv("RATE_LIMIT_PER_IP", "100"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    # ---- 数据目录 ----
    DATA_DIR: Path = BASE_DIR / "data"
    KNOWLEDGE_DIR: Path = DATA_DIR / "knowledge"
    WEB_DIR: Path = BASE_DIR / "web"


settings = Settings()
