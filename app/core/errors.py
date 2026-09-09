"""统一错误码与异常处理（对应需求分析 v2 第七节 错误码规范）

错误码定义：
- 0      成功
- 1001   参数错误（400）
- 1002   未认证（401）
- 1003   无权限（403）
- 2001   资源不存在（404）
- 3001   请求过于频繁（429）
- 4001   服务内部错误（500）
- 4002   LLM 服务不可用（503，降级到关键词匹配）
- 4003   LLM 响应超时（504）

统一响应结构：{"code": int, "message": str, "data": null}
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..utils.helpers import fail
from .logger import logger


class BizError(Exception):
    """业务异常：携带错误码与 HTTP 状态码"""

    def __init__(self, code: int, message: str, status_code: int = 400, data=None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.data = data
        super().__init__(message)


class BadRequestError(BizError):
    def __init__(self, message="参数错误", data=None):
        super().__init__(1001, message, status_code=400, data=data)


class UnauthorizedError(BizError):
    def __init__(self, message="未认证或登录已过期", data=None):
        super().__init__(1002, message, status_code=401, data=data)


class ForbiddenError(BizError):
    def __init__(self, message="无权限访问该资源", data=None):
        super().__init__(1003, message, status_code=403, data=data)


class NotFoundError(BizError):
    def __init__(self, message="资源不存在", data=None):
        super().__init__(2001, message, status_code=404, data=data)


class RateLimitError(BizError):
    def __init__(self, message="请求过于频繁，请稍后重试", retry_after: int = 60, data=None):
        super().__init__(3001, message, status_code=429, data=data)
        self.retry_after = retry_after


class LLMUnavailableError(BizError):
    def __init__(self, message="大模型服务暂不可用，已降级到本地知识库", data=None):
        super().__init__(4002, message, status_code=503, data=data)


class LLMTimeoutError(BizError):
    def __init__(self, message="大模型响应超时，请稍后重试", data=None):
        super().__init__(4003, message, status_code=504, data=data)


# 错误码 → HTTP 状态码（用于文档/日志）
ERROR_CODES = {
    0: 200, 1001: 400, 1002: 401, 1003: 403,
    2001: 404, 3001: 429, 4001: 500, 4002: 503, 4003: 504,
}


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器，统一错误响应格式"""

    @app.exception_handler(BizError)
    async def _handle_biz(request: Request, exc: BizError):
        headers = {}
        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content=fail(exc.code, exc.message, exc.data),
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception):
        logger.exception("未捕获异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content=fail(4001, "服务内部错误，请稍后重试", None),
        )
