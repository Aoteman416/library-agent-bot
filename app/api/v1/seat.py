"""座位预约接口（JWT 鉴权，对应 v2 权限矩阵）"""
from fastapi import APIRouter, Depends, Query

from ...core.auth import CurrentUser, get_current_user
from ...core.errors import UnauthorizedError
from ...core.logger import logger
from ...models.database import record_operation
from ...models.schemas import SeatReserveRequest
from ...services.seat_service import seat_service
from ...utils.helpers import fail, ok


router = APIRouter(prefix="/seats", tags=["座位"])


@router.get("/availability")
async def availability():
    """座位余量（匿名可访问）"""
    return ok(seat_service.get_availability())


@router.post("/reserve")
async def reserve(payload: SeatReserveRequest, current_user: CurrentUser = Depends(get_current_user)):
    """预约座位（需登录）"""
    if current_user.is_guest:
        raise UnauthorizedError("座位预约需先登录")
    result = seat_service.reserve(
        payload.user_id, payload.area, payload.date, payload.start_time, payload.end_time
    )
    if not result["ok"]:
        return fail(400, result["message"])
    record_operation(payload.user_id, "seat_reserve", "seat", payload.area)
    return ok(result, message=result["message"])


@router.post("/cancel")
async def cancel(reservation_id: str = Query(..., min_length=1), current_user: CurrentUser = Depends(get_current_user)):
    """取消预约（需登录）"""
    if current_user.is_guest:
        raise UnauthorizedError("取消预约需先登录")
    result = seat_service.cancel(reservation_id)
    if not result["ok"]:
        return fail(404, result["message"])
    return ok(None, message=result["message"])


@router.get("/reservations")
async def reservations(current_user: CurrentUser = Depends(get_current_user)):
    """我的预约（需登录）"""
    if current_user.is_guest:
        raise UnauthorizedError("查询预约需先登录")
    return ok(seat_service.list_reservations(current_user.user_id))
