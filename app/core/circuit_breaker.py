"""熔断降级（对应流程图 Module 2.7）

为三个下游分别提供独立熔断器：
- 大模型（llm_breaker）    ：连续失败 5 次打开，30s 后恢复
- 知识库（kb_breaker）     ：连续 5 次失败打开，30s 后恢复
- 借阅系统（borrow_breaker）：连续 3 次失败打开，20s 后恢复

熔断器打开时调用方应降级（模板/离线/人工），而不是把故障抛给用户。
提供三层原语：is_open()（试探前判断）、record_failure()、record_success()；
以及便捷包装 guard_async / guard_sync（自动记录并在打开时抛 DownstreamError）。
"""
import time
from enum import Enum
from typing import Awaitable, Callable, Optional

from ..config import settings
from .logger import logger


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """极简熔断器：基于连续失败计数 + 恢复超时"""

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        recovery_seconds: int,
        enabled: bool = True,
        half_open_required: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.enabled = enabled
        self._half_open_required = half_open_required
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_successes = 0

    def is_open(self) -> bool:
        """调用前判断：是否应被拦截（打开/半开超时后允许试探）"""
        if not self.enabled:
            return False
        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.recovery_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                logger.info("熔断器[%s] 进入半开状态，允许一次试探", self.name)
                return False
            return True
        return False

    def record_failure(self) -> None:
        if not self.enabled:
            return
        self._failures += 1
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._failures >= self.failure_threshold:
            self._trip()

    def record_success(self) -> None:
        if not self.enabled:
            return
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_required:
                self._state = CircuitState.CLOSED
                self._failures = 0
                logger.info("熔断器[%s] 恢复闭合", self.name)
        else:
            self._failures = 0

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        logger.warning(
            "熔断器[%s] 打开，下游调用将被拦截约 %ss", self.name, self.recovery_seconds
        )

    @property
    def state(self) -> str:
        return self._state.value

    # ---- 便捷包装：打开时抛 DownstreamError ----
    async def guard_async(self, coro: Awaitable):
        if self.is_open():
            from .errors import DownstreamError

            raise DownstreamError(f"{self.name} 服务暂时不可用（熔断），请稍后重试")
        try:
            result = await coro
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def guard_sync(self, func: Callable, *args, **kwargs):
        if self.is_open():
            from .errors import DownstreamError

            raise DownstreamError(f"{self.name} 服务暂时不可用（熔断），请稍后重试")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


# ================= 三个下游熔断器（按配置阈值）=================
llm_breaker = CircuitBreaker(
    "大模型",
    settings.CB_LLM_FAILURE_THRESHOLD,
    settings.CB_LLM_RECOVERY_SECONDS,
    enabled=settings.CIRCUIT_BREAKER_ENABLED,
)
kb_breaker = CircuitBreaker(
    "知识库",
    settings.CB_KB_FAILURE_THRESHOLD,
    settings.CB_KB_RECOVERY_SECONDS,
    enabled=settings.CIRCUIT_BREAKER_ENABLED,
)
borrow_breaker = CircuitBreaker(
    "借阅系统",
    settings.CB_BORROW_FAILURE_THRESHOLD,
    settings.CB_BORROW_RECOVERY_SECONDS,
    enabled=settings.CIRCUIT_BREAKER_ENABLED,
)
