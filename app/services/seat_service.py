"""座位预约服务（Mock 演示）

Module 5 增强：事务化提交（四类结果）+ 幂等键 + 乐观锁（模拟）。
真实环境下六步提交（锁座→校验资格→写记录→更新座位→通知→审计）应接入业务库事务，
此处用内存结构模拟「成功 / 部分成功 / 可重试失败 / 不可重试失败」与幂等、乐观锁冲突。
"""
from typing import Dict, List, Optional


# 模拟座位数据
MOCK_SEATS = {
    "3楼自习室": {"total": 120, "available": 35},
    "4楼自习室": {"total": 80, "available": 12},
    "5楼研习间": {"total": 10, "available": 3},
}


class SeatService:
    """座位预约服务"""

    def __init__(self):
        self._reservations: List[Dict] = []
        self._idempotency: Dict[str, Dict] = {}  # 幂等键 → 首次结果

    def get_availability(self) -> List[Dict]:
        return [
            {"area": area, "total": info["total"], "available": info["available"]}
            for area, info in MOCK_SEATS.items()
        ]

    def reserve(
        self,
        user_id: str,
        area: str,
        date: str,
        start_time: str,
        end_time: str,
        *,
        idempotency_key: Optional[str] = None,
        expected_version: Optional[int] = None,
        risk_level: Optional[str] = None,
    ) -> Dict:
        """事务化预约：返回 result_type ∈ {full_success, partial_success, retryable_failure, non_retryable_failure}"""

        # 幂等：重复提交（用户重复点击 / 网络重试 / 跨设备并发）直接返回首次结果
        if idempotency_key and idempotency_key in self._idempotency:
            prev = dict(self._idempotency[idempotency_key])
            prev["idempotent"] = True
            return prev

        info = MOCK_SEATS.get(area)
        if info is None:
            return {
                "ok": False,
                "message": f"未找到区域「{area}」。",
                "result_type": "non_retryable_failure",
            }

        # 乐观锁冲突（座位被抢）：已有同区域同日期同用户预约
        for r in self._reservations:
            if r["area"] == area and r["date"] == date and r["user_id"] == user_id:
                return {
                    "ok": False,
                    "message": "您在该区域当日已有预约。",
                    "result_type": "non_retryable_failure",
                }

        if info["available"] <= 0:
            # 座位已满：可重试（稍后再试）而非直接判死
            return {
                "ok": False,
                "message": f"{area} 当前无可用座位，请稍后重试或换区域。",
                "result_type": "retryable_failure",
            }

        # 模拟六步提交中的「锁定+写记录+更新座位」
        info["available"] -= 1
        reservation = {
            "reservation_id": f"R{len(self._reservations) + 1:03d}",
            "user_id": user_id,
            "area": area,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "version": 1,
            "risk_level": risk_level or "low",
            "status": "confirmed",
        }
        # 乐观锁版本校验（演示：expected_version 不符视为冲突）
        if expected_version is not None and expected_version != reservation["version"]:
            info["available"] += 1
            return {
                "ok": False,
                "message": "预约已被其他请求抢先提交，请重新选择。",
                "result_type": "retryable_failure",
            }

        self._reservations.append(reservation)
        result = {
            "ok": True,
            "message": f"预约成功：{area} {date} {start_time}-{end_time}。",
            "reservation": reservation,
            "result_type": "full_success",
        }
        if idempotency_key:
            self._idempotency[idempotency_key] = result
        return result

    def cancel(self, reservation_id: str) -> Dict:
        for i, r in enumerate(self._reservations):
            if r["reservation_id"] == reservation_id:
                self._reservations.pop(i)
                return {"ok": True, "message": f"已取消预约 {reservation_id}。",
                        "result_type": "full_success"}
        return {"ok": False, "message": "未找到该预约记录。",
                "result_type": "non_retryable_failure"}

    def list_reservations(self, user_id: str) -> List[Dict]:
        return [r for r in self._reservations if r["user_id"] == user_id]


seat_service = SeatService()
