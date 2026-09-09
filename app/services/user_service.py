"""用户服务：认证、借阅管理

设计要点（对应需求分析 v2 5.1 / 9.1）：
- 认证优先用关系数据库 users 表校验用户存在性与角色（DB 可用时）
- 演示密码仍由 Mock 映射提供（真实环境应改为 users.token_hash 比对）
- DB 不可用时完全回退 Mock，保证开箱即用
- 管理员演示凭证来自 settings.ADMIN_USER_ID / ADMIN_PASSWORD
"""
from typing import Dict, Optional, Tuple

from ..config import settings
from ..models.database import get_user


# 模拟用户数据（演示用，真实环境密码哈希存 users 表）
MOCK_USERS = {
    "2024001": {"name": "张三", "role": "student", "password": "123456"},
    "2024002": {"name": "李四", "role": "student", "password": "123456"},
}

# 模拟借阅记录
MOCK_BORROWS = {
    "2024001": [{"book_id": "三体", "title": "三体", "due": "2026-07-01", "renew_count": 0}],
    "2024002": [
        {"book_id": "C++ Primer 第六版", "title": "C++ Primer 第六版", "due": "2026-06-15", "renew_count": 1},
        {"book_id": "Python编程：从入门到实践", "title": "Python编程：从入门到实践", "due": "2026-08-30", "renew_count": 0},
    ],
}

MOCK_FINES = {"2024001": 0.0, "2024002": 2.5}


class UserService:
    def authenticate(self, user_id: str, password: str) -> Tuple[bool, str, Dict]:
        """认证：返回 (是否成功, 消息, 用户信息)"""
        # 管理员演示凭证
        if user_id == settings.ADMIN_USER_ID and password == settings.ADMIN_PASSWORD:
            return True, "管理员登录成功", {
                "user_id": user_id, "name": "管理员", "role": "admin",
            }
        user = MOCK_USERS.get(user_id)
        if user and user["password"] == password:
            role = user["role"]
            # 若关系库可用，以库内角色为准
            db_user = get_user(user_id)
            if db_user:
                role = db_user["user_type"]
            return True, f"欢迎你，{user['name']}同学！已登录为学号 {user_id}", {
                "user_id": user_id, "name": user["name"], "role": role,
            }
        return False, "学号或密码错误", {}

    def get_borrow_info(self, user_id: str) -> Optional[Dict]:
        if user_id not in MOCK_USERS and get_user(user_id) is None:
            return None
        return {
            "user_id": user_id,
            "borrowed": MOCK_BORROWS.get(user_id, []),
            "fines": MOCK_FINES.get(user_id, 0.0),
        }

    def renew(self, user_id: str, book_id: str) -> Dict:
        """续借"""
        borrows = MOCK_BORROWS.get(user_id, [])
        book = next((b for b in borrows if book_id in b["book_id"] or book_id in b["title"]), None)
        if book is None:
            return {"ok": False, "message": "未找到该书的借阅记录。"}
        if book["renew_count"] >= 1:
            return {"ok": False, "message": "该书已续借过，无法再次续借。"}
        book["renew_count"] += 1
        from datetime import date, timedelta
        due = date.fromisoformat(book["due"]) + timedelta(days=30)
        book["due"] = due.isoformat()
        return {
            "ok": True,
            "message": f"《{book['title']}》续借成功，新到期日 {book['due']}。",
            "new_due": book["due"],
        }


user_service = UserService()
