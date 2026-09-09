"""后台管理接口：统计 / 知识库维护 / 反馈管理 / 知识回流

权限：管理类接口需管理员角色；演示模式（DEBUG=true）下对匿名放行，便于本地查看。
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...config import settings
from ...core.auth import CurrentUser, get_current_user
from ...core.errors import ForbiddenError
from ...core.logger import logger
from ...services.admin_service import admin_service
from ...services.knowledge_service import knowledge_service
from ...services.rag_service import rag_service
from ...utils.helpers import fail, ok


router = APIRouter(prefix="/admin", tags=["管理"])


def _require_admin(current_user: CurrentUser) -> None:
    """管理员校验：演示模式（DEBUG）对匿名放行，部署环境严格鉴权"""
    if current_user.is_admin:
        return
    if settings.DEBUG:
        return
    raise ForbiddenError("此功能需要管理员权限")


class KbDocumentIn(BaseModel):
    """新增知识库文档（纯文本，避免上传依赖）"""
    file_name: str = Field(..., min_length=1, max_length=100, description="文件名，如 借阅规则.md")
    content: str = Field(..., min_length=1, description="文档内容（Markdown）")


class ResolveUnresolvedIn(BaseModel):
    """馆员补充答案"""
    index: int = Field(..., description="未解决问题索引")
    answer: str = Field(..., min_length=1, description="馆员提供的答案")


class KnowledgeRefluxIn(BaseModel):
    """知识回流：将待回流条目写入知识库"""
    source_index: int = Field(..., description="待回流条目索引")
    file_name: str = Field(..., description="写入的知识库文件名")


# ==================== 统计接口 ====================

@router.get("/stats/overview")
async def stats_overview(current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    return ok(admin_service.get_overview())


@router.get("/stats/channels")
async def stats_channels(current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    return ok({
        "distribution": admin_service.get_channel_distribution(),
        "percentage": admin_service.get_overview().get("channel_percentage", {}),
    })


@router.get("/stats/unresolved")
async def stats_unresolved(
    status: Optional[str] = Query(None, description="按状态筛选: pending/resolved"),
    current_user: CurrentUser = Depends(get_current_user),
):
    _require_admin(current_user)
    items = admin_service.get_unresolved(status=status)
    return ok({"count": len(items), "questions": items})


@router.get("/stats/feedbacks")
async def stats_feedbacks(
    limit: int = Query(50, ge=1, le=500),
    current_user: CurrentUser = Depends(get_current_user),
):
    _require_admin(current_user)
    return ok({"count": len(admin_service._feedbacks), "feedbacks": admin_service.get_feedbacks(limit)})


@router.get("/stats/knowledge-pending")
async def stats_knowledge_pending(current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    return ok({"count": len(admin_service.get_pending_knowledge()), "items": admin_service.get_pending_knowledge()})


# ==================== 知识回流闭环 ====================

@router.post("/unresolved/resolve")
async def resolve_unresolved(payload: ResolveUnresolvedIn, current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    success = admin_service.resolve_unresolved(payload.index, payload.answer)
    if not success:
        return fail(404, "未找到对应的未解决问题")
    return ok({"resolved": True, "index": payload.index}, message="已标记为解决并加入待回流队列")


@router.post("/kb/reflux")
async def knowledge_reflux(payload: KnowledgeRefluxIn, current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    pending = admin_service.get_pending_knowledge()
    if not (0 <= payload.source_index < len(pending)):
        return fail(404, "待回流条目不存在")

    item = pending[payload.source_index]
    content = f"## {item.get('question', '待补充知识')}\n\n{item.get('answer', item.get('comment', ''))}\n\n<!-- 来源: {item.get('source', 'unknown')} | 时间: {item.get('timestamp', '')} -->"

    safe_name = Path(payload.file_name).name
    if not safe_name.lower().endswith((".md", ".txt")):
        return fail(400, "仅支持 .md / .txt 文档")
    target = settings.KNOWLEDGE_DIR / safe_name
    target.write_text(content, encoding="utf-8")
    rag_service.build_index(force_rebuild=True)
    logger.info("知识回流：条目 %d → %s，索引已重建", payload.source_index, safe_name)
    return ok(
        {"file_name": safe_name, "chunks": rag_service.kb_size},
        message="知识已回流并重建索引",
    )


# ==================== 知识库维护 ====================

@router.get("/kb/info")
async def kb_info(current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    return ok({
        "documents": knowledge_service.documents,
        "chunk_count": knowledge_service.chunks.__len__(),
        "indexed": rag_service.indexed,
        "index_size": rag_service.kb_size,
        "mode": settings.EMBEDDING_MODE,
    })


@router.post("/kb/document")
async def add_document(payload: KbDocumentIn, current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    safe_name = Path(payload.file_name).name
    if not safe_name.lower().endswith((".md", ".txt")):
        return fail(400, "仅支持 .md / .txt 文档")
    target = settings.KNOWLEDGE_DIR / safe_name
    target.write_text(payload.content, encoding="utf-8")
    rag_service.build_index(force_rebuild=True)
    logger.info("新增知识库文档：%s，重建索引完成", safe_name)
    return ok(
        {
            "file_name": safe_name,
            "chunks": rag_service.kb_size,
            "indexed": rag_service.indexed,
        },
        message="知识库已更新",
    )


@router.post("/kb/reindex")
async def reindex(current_user: CurrentUser = Depends(get_current_user)):
    _require_admin(current_user)
    rag_service.build_index(force_rebuild=True)
    return ok({"index_size": rag_service.kb_size}, message="索引重建完成")
