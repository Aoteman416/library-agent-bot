"""鉴权接口：刷新 token / 当前用户（对应需求分析 v2 9.1 JWT）"""
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...core.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
)
from ...core.errors import UnauthorizedError
from ...utils.helpers import ok


router = APIRouter(prefix="/auth", tags=["鉴权"])


class RefreshRequest(BaseModel):
    refresh_token: str = ""


@router.post("/refresh")
async def refresh(payload: RefreshRequest):
    """用 Refresh Token 换取新的 Access Token（旧 refresh 一并轮换）"""
    if not payload.refresh_token:
        raise UnauthorizedError("请提供 refresh_token")
    data = decode_token(payload.refresh_token)
    if data.get("type") != "refresh":
        raise UnauthorizedError("请提供 refresh token（而非 access token）")
    user_id = data.get("sub", "guest")
    role = data.get("role", "student")
    access = create_access_token(user_id, role)
    new_refresh = create_refresh_token(user_id, role)
    return ok({
        "access_token": access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    })


@router.get("/me")
async def me(current_user: CurrentUser = Depends(get_current_user)):
    """返回当前登录用户信息（未登录返回 1002）"""
    if current_user.is_guest:
        raise UnauthorizedError()
    return ok({
        "user_id": current_user.user_id,
        "role": current_user.role,
    })
