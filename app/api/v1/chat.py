"""对话接口 V2：使用新的 ChatServiceV2 + 反馈 + 升级"""
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...core.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    get_current_user,
)
from ...core.logger import logger
from ...core.permissions import auth_upgrade_flow, permission_manager
from ...core.session_manager import session_manager
from ...models.schemas import ChatMessage, FeedbackRequest, LoginRequest
from ...services.chat_service import chat_service
from ...services.user_service import user_service
from ...utils.helpers import fail, ok, sse_event

router = APIRouter(prefix="/chat", tags=["对话"])

# 存储用户反馈（内存）
_feedback_store: list = []


@router.post("/stream")
async def chat_stream(
    payload: ChatMessage,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """SSE 流式对话（V2）"""
    from asyncio import CancelledError

    user_id = current_user.user_id
    user_role = current_user.role
    session = session_manager.get_or_create(
        payload.session_id, user_id,
        is_public=request.headers.get("X-Public-Device", "false") == "true",
    )

    async def event_generator():
        try:
            yield sse_event(
                "session",
                {"session_id": session.session_id, "phase": session.state.phase},
            )
            full_content = ""
            async for ev in chat_service.process(session, payload.message, user_id, user_role):
                if ev["event"] == "token":
                    full_content += ev["data"].get("content", "")
                yield sse_event(ev["event"], ev["data"])
        except CancelledError:
            logger.warning("流式对话中断: session=%s, content_len=%d", session.session_id, len(full_content))
            yield sse_event("partial", {
                "reason": "client_disconnected",
                "partial_content": full_content[:200],
                "session_id": session.session_id,
            })
        except Exception as exc:
            logger.exception("流式对话异常")
            yield sse_event("error", {"code": "INTERNAL", "message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("")
async def chat(
    payload: ChatMessage,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """非流式对话接口"""
    user_id = current_user.user_id
    user_role = current_user.role
    session = session_manager.get_or_create(payload.session_id, user_id)

    result = {}
    captured_channel = "unknown"
    async for ev in chat_service.process(session, payload.message, user_id, user_role):
        if ev["event"] == "route":
            captured_channel = ev["data"].get("channel", "unknown")
        if ev["event"] == "done":
            result = ev["data"]
    return ok({
        "reply_type": result.get("reply_type", "text"),
        "content": result.get("answer", ""),
        "sources": result.get("sources"),
        "channel": result.get("channel", captured_channel),
        "session_id": session.session_id,
        "session_phase": session.state.phase,
        "security_blocked": result.get("security_blocked", False),
        "low_confidence": result.get("low_confidence", False),
    })


@router.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    """用户反馈：满意度评分/改进建议（用于知识库回流、评测数据收集）"""
    from ...services.admin_service import admin_service

    feedback = {
        "feedback_id": __import__("uuid").uuid4().hex[:10],
        "session_id": payload.session_id,
        "user_id": request.headers.get("X-Demo-User", "guest"),
        "rating": payload.rating,
        "comment": payload.comment or "",
        "message": payload.message or "",
        "is_helpful": payload.is_helpful,
        "submitted_at": datetime.now().isoformat(),
    }
    _feedback_store.append(feedback)

    admin_service.submit_feedback(
        session_id=payload.session_id,
        rating=payload.rating,
        comment=payload.comment or "",
        is_helpful=payload.is_helpful,
    )
    return ok({"status": "received", "feedback_id": feedback["feedback_id"]})


@router.get("/feedback/stats")
async def feedback_stats():
    """反馈统计：用于评估 RAG 质量"""
    total = len(_feedback_store)
    if not total:
        return ok({"total": 0, "avg_rating": 0, "helpful_rate": 0})
    ratings = [f["rating"] for f in _feedback_store if f["rating"]]
    helpful = [f for f in _feedback_store if f.get("is_helpful")]
    return ok({
        "total": total,
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0,
        "helpful_rate": round(len(helpful) / total * 100, 1),
        "low_rating_count": sum(1 for r in ratings if r <= 2),
    })


@router.post("/login")
async def login(payload: LoginRequest):
    """登录接口：签发 JWT（access + refresh）"""
    auth_ok, message, info = user_service.authenticate(payload.user_id, payload.password)
    if not auth_ok:
        return fail(401, message)
    access = create_access_token(info["user_id"], info["role"])
    refresh = create_refresh_token(info["user_id"], info["role"])
    return ok({
        "user_id": info["user_id"],
        "name": info.get("name", ""),
        "role": info["role"],
        "token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "message": message,
    })


@router.post("/login/complete-upgrade")
async def complete_upgrade(payload: Dict, request: Request):
    """登录升级完成：执行挂起的业务请求"""
    session_id = payload.get("session_id", "")
    user_id = payload.get("user_id", "")
    result = auth_upgrade_flow.complete_upgrade(session_id, user_id)
    return ok(result)


@router.get("/permissions/matrix")
async def get_permission_matrix():
    """权限矩阵查询"""
    return ok(permission_manager.get_matrix())


@router.get("/sessions")
async def list_sessions():
    return ok(session_manager.list_sessions())


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    session = session_manager.get(session_id)
    if session is None:
        return fail(404, "会话不存在")
    return ok({
        "session_id": session.session_id,
        "user_id": session.user_id,
        "phase": session.state.phase,
        "intent": session.state.intent,
        "created_at": session.created_at.isoformat(),
        "messages": session.history(),
    })


@router.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """清空会话（公共设备使用）"""
    session = session_manager.get(session_id)
    if session is None:
        return fail(404, "会话不存在")
    session.clear()
    return ok({"status": "cleared"})
