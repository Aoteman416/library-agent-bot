"""人工转接 / 工单队列（Module 9）

轻量实现：内存工单队列 + 排队位置 / 预计等待；并触发「学习回流」（未解决 / 转人工记录）。
真实环境应接企业微信 / 飞书客服，或自建 human_tickets 表 + 状态机（受理 → 处理中 → 已解决）。
"""
import time
import uuid
from typing import Dict, List, Optional

from ..core.logger import logger


class HumanTransferService:
    """人工转接工单服务"""

    def __init__(self):
        self._tickets: List[Dict] = []
        self._queue: List[str] = []  # 待处理 ticket_id 队列

    def create_ticket(
        self, user_id: str, session_id: str, message: str, intent: str = "human_transfer"
    ) -> Dict:
        ticket_id = f"H{uuid.uuid4().hex[:8].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "intent": intent,
            "status": "queued",
            "created_at": time.time(),
        }
        self._tickets.append(ticket)
        self._queue.append(ticket_id)
        position = self._queue.index(ticket_id) + 1
        estimated_wait = f"{max(1, position * 2)}分钟"

        # 学习回流：转人工视为需跟进的未解决，沉淀到运营数据
        try:
            from .admin_service import admin_service

            admin_service.record_unresolved(message, session_id=session_id, confidence=0.0)
        except Exception:  # noqa: BLE001
            pass

        logger.info("人工转接工单创建: ticket=%s pos=%d user=%s", ticket_id, position, user_id)
        return {
            "ticket_id": ticket_id,
            "queue_position": position,
            "estimated_wait": estimated_wait,
            "status": "queued",
        }

    def pending_count(self) -> int:
        return len(self._queue)

    def list_tickets(self, limit: int = 50) -> List[Dict]:
        return self._tickets[-limit:]


human_service = HumanTransferService()
