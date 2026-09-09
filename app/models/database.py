"""关系数据库模型与连接（对应需求分析 v2 第五节 5.1）

设计要点：
- 表：users / sessions / messages / knowledge_documents / unresolved_questions / operation_logs
- 默认 DB_ENABLED=false：使用内存 SQLite 演示（自动建表 + 灌 Demo 用户），开箱即用
- DB_ENABLED=true 且 DATABASE_URL 指向 MySQL：连接真实库（pymysql 驱动）
- 全程可选依赖：sqlalchemy 缺失时仅标记不可用，service 层自动回退 Mock，不报错

注：v2 5.1 未定义 borrow_records / seats 表，借阅与座位在演示中由 service 层 Mock 提供；
users 表用于鉴权，operation_logs 用于操作审计，sessions/messages 用于对话持久化。
"""
from typing import Iterator, Optional

from ..config import settings


class DBUnavailable(Exception):
    """数据库不可用（service 层据此回退 Mock）"""


try:
    from sqlalchemy import (  # noqa: F401
        Column,
        String,
        Integer,
        Text,
        DateTime,
        JSON,
        Index,
        func,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker, Session
    _HAS_SQLA = True
except ImportError:  # pragma: no cover - 缺依赖时降级
    _HAS_SQLA = False

Base = declarative_base() if _HAS_SQLA else object


if _HAS_SQLA:
    class User(Base):
        __tablename__ = "users"
        user_id = Column(String(32), primary_key=True)
        username = Column(String(50), nullable=False)
        user_type = Column(String(20), default="student")
        token_hash = Column(String(128), default="")
        created_at = Column(DateTime, server_default=func.now())
        updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    class SessionModel(Base):
        __tablename__ = "sessions"
        session_id = Column(String(32), primary_key=True)
        user_id = Column(String(32), index=True)
        title = Column(String(100), default="")
        created_at = Column(DateTime, server_default=func.now())
        updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
        __table_args__ = (Index("idx_user_id", "user_id"),)

    class Message(Base):
        __tablename__ = "messages"
        message_id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(32), index=True)
        role = Column(String(20))
        content = Column(Text)
        token_count = Column(Integer, default=0)
        sources = Column(JSON, default=None)
        created_at = Column(DateTime, server_default=func.now())
        __table_args__ = (Index("idx_msg_session", "session_id"),)

    class KnowledgeDocument(Base):
        __tablename__ = "knowledge_documents"
        doc_id = Column(String(32), primary_key=True)
        file_name = Column(String(255), nullable=False)
        file_type = Column(String(20), default="md")
        category = Column(String(50), default="")
        chunk_count = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        uploaded_by = Column(String(32), default="")
        created_at = Column(DateTime, server_default=func.now())

    class UnresolvedQuestion(Base):
        __tablename__ = "unresolved_questions"
        id = Column(Integer, primary_key=True, autoincrement=True)
        question = Column(Text, nullable=False)
        session_id = Column(String(32), default="")
        user_id = Column(String(32), default="")
        intent_predicted = Column(String(50), default="")
        resolved = Column(Integer, default=0)
        created_at = Column(DateTime, server_default=func.now())
        __table_args__ = (Index("idx_unresolved", "resolved"),)

    class OperationLog(Base):
        __tablename__ = "operation_logs"
        log_id = Column(Integer, primary_key=True, autoincrement=True)
        user_id = Column(String(32), default="")
        action = Column(String(50), nullable=False)
        target_type = Column(String(50), default="")
        target_id = Column(String(32), default="")
        detail = Column(JSON, default=None)
        created_at = Column(DateTime, server_default=func.now())
        __table_args__ = (Index("idx_op_user", "user_id", "action"),)


# 引擎与会话工厂单例
_engine = None
_SessionLocal = None
_db_available = False


def init_db() -> bool:
    """初始化数据库：建表 + 演示数据。返回是否可用"""
    global _engine, _SessionLocal, _db_available
    if not _HAS_SQLA:
        return False
    try:
        if settings.DB_ENABLED:
            url = settings.DATABASE_URL
        elif settings.DB_DEMO_MODE:
            url = "sqlite:///:memory:"
        else:
            return False
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        _engine = create_engine(url, pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, future=True)
        Base.metadata.create_all(_engine)
        if not settings.DB_ENABLED and settings.DB_DEMO_MODE:
            _seed_demo(_SessionLocal())
        _db_available = True
        return True
    except Exception as exc:  # noqa: BLE001
        from .logger import logger
        logger.warning("数据库初始化失败，业务数据回退 Mock：%s", exc)
        _db_available = False
        return False


def _seed_demo(db: "Session") -> None:
    """向内存演示库灌入 Demo 用户（与 Mock 凭证一致）"""
    demo_users = [
        User(user_id="2024001", username="张三", user_type="student"),
        User(user_id="2024002", username="李四", user_type="student"),
        User(user_id=settings.ADMIN_USER_ID, username="管理员", user_type="admin"),
    ]
    for u in demo_users:
        db.add(u)
    db.commit()


def get_db() -> Iterator["Session"]:
    """FastAPI 依赖：提供数据库会话，自动关闭"""
    if not _db_available or _SessionLocal is None:
        raise DBUnavailable("数据库未启用")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_available() -> bool:
    """数据库是否可用（已成功初始化并建表）"""
    return _db_available


def get_user(user_id: str) -> Optional[dict]:
    """按 user_id 查询用户（返回纯 dict，便于 service 层使用）"""
    if not _db_available or _SessionLocal is None:
        return None
    with _SessionLocal() as db:
        row = db.query(User).filter(User.user_id == user_id).first()
        if not row:
            return None
        return {
            "user_id": row.user_id,
            "username": row.username,
            "user_type": row.user_type,
        }


def record_operation(user_id: str, action: str, target_type: str = "", target_id: str = "", detail=None) -> None:
    """写入操作日志（DB 不可用时静默跳过）"""
    if not _db_available or _SessionLocal is None:
        return
    try:
        with _SessionLocal() as db:
            db.add(OperationLog(
                user_id=user_id, action=action,
                target_type=target_type, target_id=target_id or "",
                detail=detail,
            ))
            db.commit()
    except Exception as exc:  # noqa: BLE001
        from .logger import logger
        logger.warning("操作日志写入失败（忽略）：%s", exc)


def persist_message(session_id: str, role: str, content: str, sources=None, token_count: int = 0) -> None:
    """持久化消息（DB 不可用时静默跳过）"""
    if not _db_available or _SessionLocal is None:
        return
    try:
        with _SessionLocal() as db:
            db.add(Message(
                session_id=session_id, role=role,
                content=content, sources=sources, token_count=token_count,
            ))
            db.commit()
    except Exception as exc:  # noqa: BLE001
        from .logger import logger
        logger.warning("消息持久化失败（忽略）：%s", exc)
