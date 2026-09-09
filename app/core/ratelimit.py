"""限流中间件（对应需求分析 v2 9.1 rate_limiting）

设计要点：
- 双维度计数：按用户（Bearer sub / X-Demo-User）与按客户端 IP
- 滑动窗口：默认 60s 内 per_user=30 / per_ip=100（可调）
- 优先使用 Redis（REDIS_ENABLED 且客户端可用）做分布式计数，否则进程内计数
- 超限返回 429（错误码 3001，响应头带 Retry-After），不抛异常保证链路稳定
"""
import time
from collections import defaultdict, deque
from typing import Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import settings
from ..utils.helpers import fail
from .logger import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于滑动窗口的限流中间件"""

    def __init__(self, app, per_user: int = None, per_ip: int = None, window: int = None):
        super().__init__(app)
        self.per_user = per_user or settings.RATE_LIMIT_PER_USER
        self.per_ip = per_ip or settings.RATE_LIMIT_PER_IP
        self.window = window or settings.RATE_LIMIT_WINDOW_SECONDS
        self._memory: Dict[str, deque] = defaultdict(deque)
        self._redis = None
        if settings.REDIS_ENABLED:
            try:
                import redis  # noqa: F401
                self._redis = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    socket_connect_timeout=0.5,
                )
                self._redis.ping()
                logger.info("限流中间件使用 Redis 后端 (%s:%s)", settings.REDIS_HOST, settings.REDIS_PORT)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis 不可用，限流回退进程内计数：%s", exc)
                self._redis = None

    def _client_ip(self, request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _user_key(self, request) -> str:
        """返回用户维度的限流 key；匿名则用 IP 维度"""
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            from .auth import decode_token
            try:
                return f"user:{decode_token(auth.split(' ', 1)[1].strip()).get('sub', 'anon')}"
            except Exception:
                pass
        demo = request.headers.get("X-Demo-User")
        if demo:
            return f"user:{demo}"
        return f"ip:{self._client_ip(request)}"

    async def dispatch(self, request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path
        if path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        now = time.time()
        # 用户维度
        if not self._allow(self._user_key(request), self.per_user, now):
            return self._too_many()
        # IP 维度
        if not self._allow(f"ip:{self._client_ip(request)}", self.per_ip, now):
            return self._too_many()

        return await call_next(request)

    def _too_many(self) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content=fail(3001, "请求过于频繁，请稍后重试", None),
            headers={"Retry-After": str(self.window)},
        )

    def _allow(self, key: str, limit: int, now: float) -> bool:
        if self._redis:
            try:
                rkey = f"ratelimit:{key}"
                cnt = self._redis.incr(rkey)
                if cnt == 1:
                    self._redis.expire(rkey, self.window)
                return cnt <= limit
            except Exception:  # noqa: BLE001
                pass
        dq = self._memory[key]
        while dq and dq[0] <= now - self.window:
            dq.popleft()
        dq.append(now)
        return len(dq) <= limit
