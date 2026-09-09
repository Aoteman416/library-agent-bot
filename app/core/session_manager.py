"""会话管理器：升级版（含 DST + 会话隔离 + 超时控制）

解决的问题（对应审查清单 #2、#14）：
- 每个会话维护 DialogueState 用于槽位填充追踪
- 触屏一体机超时自动清除会话
- 会话数据支持脱敏存储
"""
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .dst import DialogueState
from .security import security_guard

# 会话超时配置
SESSION_TIMEOUT_SECONDS = 1800  # 30 分钟无活动自动失效
PUBLIC_DEVICE_TIMEOUT = 600     # 公共设备（如一体机）10 分钟超时

MAX_HISTORY_TURNS = 10


class Session:
    """单个对话会话"""

    def __init__(self, session_id: str, user_id: str = "guest", is_public: bool = False):
        self.session_id = session_id
        self.user_id = user_id
        self.is_public = is_public  # 是否公共设备
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages: List[Dict[str, str]] = []
        self.state = DialogueState(session_id)  # 对话状态追踪
        self._timeout = PUBLIC_DEVICE_TIMEOUT if is_public else SESSION_TIMEOUT_SECONDS

    def is_expired(self) -> bool:
        """会话是否已超时"""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > self._timeout

    def touch(self) -> None:
        """更新最后活动时间"""
        self.last_activity = datetime.now()

    def add(self, role: str, content: str) -> None:
        """添加消息（自动脱敏 + 持久化到关系数据库）"""
        self.touch()
        # 存储时脱敏敏感信息
        safe_content = security_guard.mask_sensitive_data(content)
        self.messages.append({"role": role, "content": safe_content})
        # 裁剪：保留最近 MAX_HISTORY_TURNS*2 条
        self.messages = self.messages[-(MAX_HISTORY_TURNS * 2):]
        # 持久化到 messages 表（DB 不可用时静默跳过）
        try:
            from .database import persist_message
            persist_message(self.session_id, role, safe_content)
        except Exception:  # noqa: BLE001
            pass

    def history(self) -> List[Dict[str, str]]:
        self.touch()
        return list(self.messages)

    def clear(self) -> None:
        """清空会话（公共设备使用）"""
        self.messages.clear()
        self.state.clear()
        self.touch()


class SessionManager:
    """会话管理容器"""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    @property
    def count(self) -> int:
        self._cleanup_expired()
        return len(self._sessions)

    def create(self, user_id: str = "guest", is_public: bool = False) -> Session:
        session_id = uuid.uuid4().hex[:12]
        session = Session(session_id, user_id, is_public)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        self._cleanup_expired()
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            session.clear()
        return session

    def get_or_create(
        self, session_id: Optional[str], user_id: str = "guest", is_public: bool = False
    ) -> Session:
        if session_id:
            session = self.get(session_id)
            if session:
                # 更新用户身份（登录升级时使用）
                if user_id != "guest":
                    session.user_id = user_id
                return session
        return self.create(user_id, is_public)

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        self._cleanup_expired()
        sessions = sorted(
            [s for s in self._sessions.values() if not s.is_expired()],
            key=lambda s: s.last_activity,
            reverse=True,
        )
        return [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "is_public": s.is_public,
                "phase": s.state.phase,
                "intent": s.state.intent,
                "message_count": len(s.messages),
                "last_activity": s.last_activity.isoformat(),
                "preview": s.messages[0]["content"][:30] if s.messages else "",
            }
            for s in sessions[:limit]
        ]

    def _cleanup_expired(self) -> None:
        """清理超时会话"""
        expired_ids = [
            sid for sid, s in self._sessions.items() if s.is_expired()
        ]
        for sid in expired_ids:
            self._sessions[sid].clear()
            del self._sessions[sid]


# 全局单例
session_manager = SessionManager()
