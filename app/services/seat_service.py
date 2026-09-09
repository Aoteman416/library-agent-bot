"""座位预约服务（Mock 演示）"""
from typing import Dict, List, Optional

# 模拟座位数据
MOCK_SEATS = {
    "3楼自习室": {"total": 120, "available": 35},
    "4楼自习室": {"total": 80, "available": 12},
    "5楼研习间": {"total": 10, "available": 3},
}

# 演示用预约记录
RESERVATIONS: List[Dict] = []


class SeatService:
    """座位预约服务"""

    def get_availability(self) -> Dict:
        return [
            {"area": area, "total": info["total"], "available": info["available"]}
            for area, info in MOCK_SEATS.items()
        ]

    def reserve(self, user_id: str, area: str, date: str, start_time: str, end_time: str) -> Dict:
        info = MOCK_SEATS.get(area)
        if info is None:
            return {"ok": False, "message": f"未找到区域「{area}」。"}
        if info["available"] <= 0:
            return {"ok": False, "message": f"{area} 当前无可用座位。"}
        # 演示冲突检查
        for r in RESERVATIONS:
            if (
                r["area"] == area
                and r["date"] == date
                and r["user_id"] == user_id
            ):
                return {"ok": False, "message": "您在该区域当日已有预约。"}
        info["available"] -= 1
        reservation = {
            "reservation_id": f"R{len(RESERVATIONS)+1:03d}",
            "user_id": user_id,
            "area": area,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
        }
        RESERVATIONS.append(reservation)
        return {
            "ok": True,
            "message": f"预约成功：{area} {date} {start_time}-{end_time}。",
            "reservation": reservation,
        }

    def cancel(self, reservation_id: str) -> Dict:
        for i, r in enumerate(RESERVATIONS):
            if r["reservation_id"] == reservation_id:
                RESERVATIONS.pop(i)
                return {"ok": True, "message": f"已取消预约 {reservation_id}。"}
        return {"ok": False, "message": "未找到该预约记录。"}

    def list_reservations(self, user_id: str) -> List[Dict]:
        return [r for r in RESERVATIONS if r["user_id"] == user_id]


seat_service = SeatService()
