"""用户接口：登录 / 借阅 / 续借（JWT 鉴权，对应 v2 9.1）"""
from fastapi import APIRouter, Depends, Query

from ...core.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    get_current_user,
)
from ...core.errors import UnauthorizedError
from ...core.logger import logger
from ...models.database import record_operation
from ...models.schemas import LoginRequest, RenewRequest
from ...services.user_service import user_service
from ...utils.helpers import fail, ok


router = APIRouter(prefix="/user", tags=["用户"])


@router.post("/login")
async def login(payload: LoginRequest):
    """登录：签发 JWT（access + refresh）"""
    success, message, info = user_service.authenticate(payload.user_id, payload.password)
    if not success:
        return fail(401, message)
    access = create_access_token(info["user_id"], info["role"])
    refresh = create_refresh_token(info["user_id"], info["role"])
    record_operation(info["user_id"], "login")
    return ok({
        "user_id": info["user_id"],
        "name": info.get("name", ""),
        "role": info["role"],
        "token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "message": message,
    }, message=message)


@router.get("/borrows")
async def get_borrows(current_user: CurrentUser = Depends(get_current_user)):
    """查询当前借阅（需登录）"""
    if current_user.is_guest:
        raise UnauthorizedError("查询借阅需先登录")
    info = user_service.get_borrow_info(current_user.user_id)
    if info is None:
        return fail(404, "用户不存在或未登录")
    return ok(info)


@router.post("/renew")
async def renew(payload: RenewRequest, current_user: CurrentUser = Depends(get_current_user)):
    """续借图书（confirm 二次确认流程，需登录）"""
    if current_user.is_guest:
        raise UnauthorizedError("续借需先登录")
    info = user_service.get_borrow_info(payload.user_id)
    if info is None:
        return fail(404, "用户不存在或未登录")
    result = user_service.renew(payload.user_id, payload.book_id)
    if not result["ok"]:
        return fail(400, result["message"])
    record_operation(payload.user_id, "renew", "book", payload.book_id)
    return ok(result, message=result["message"])


@router.get("/fines")
async def get_fines(current_user: CurrentUser = Depends(get_current_user)):
    """查询欠费（需登录）"""
    if current_user.is_guest:
        raise UnauthorizedError("查询欠费需先登录")
    info = user_service.get_borrow_info(current_user.user_id)
    if info is None:
        return fail(404, "用户不存在或未登录")
    return ok({"user_id": current_user.user_id, "fines": info["fines"]})
