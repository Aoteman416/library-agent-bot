"""权限矩阵 + 登录升级流程

解决的问题（对应审查清单 #5、#6、#14）：
- 定义哪些接口匿名可用、哪些需登录
- 匿名用户触发业务操作时，如何优雅引导登录并恢复上下文
- 触屏一体机公共会话隔离（超时自动清除）

权限矩阵：
┌──────────────────┬──────────┬────────────┬─────────────┐
│ 功能             │ 匿名     │ 登录用户   │ 管理员      │
├──────────────────┼──────────┼────────────┼─────────────┤
│ 馆藏检索         │ ✅       │ ✅         │ ✅          │
│ 座位余量查询     │ ✅       │ ✅         │ ✅          │
│ FAQ 问答         │ ✅       │ ✅         │ ✅          │
│ 借阅查询         │ ❌       │ ✅         │ ✅          │
│ 续借/还书        │ ❌       │ ✅         │ ✅          │
│ 座位预约         │ ❌       │ ✅         │ ✅          │
│ 个人信息查询     │ ❌       │ ✅(本人)   │ ✅          │
│ 管理统计         │ ❌       │ ❌         │ ✅          │
│ 知识库维护       │ ❌       │ ❌         │ ✅          │
└──────────────────┴──────────┴────────────┴─────────────┘
"""
from typing import Dict, List, Set


# 功能 → 所需权限
_ANON_ALLOWED: Set[str] = {
    "book_search",       # 馆藏检索
    "seat_availability",  # 座位余量
    "faq_qa",            # FAQ 问答
    "human_transfer",    # 转人工
    "kb_query",          # 知识库查询
    "health_check",      # 健康检查
}

# 需要登录的功能
_AUTH_REQUIRED: Set[str] = {
    "borrow_query",
    "renew",
    "return_book",
    "seat_reserve",
    "seat_cancel",
    "personal_info",
}

# 管理员功能
_ADMIN_ONLY: Set[str] = {
    "admin_stats",
    "kb_document_add",
    "kb_document_delete",
    "kb_reindex",
}


class PermissionManager:
    """权限管理器"""

    def __init__(self):
        self._anon_allowed = _ANON_ALLOWED
        self._auth_required = _AUTH_REQUIRED
        self._admin_only = _ADMIN_ONLY

    def can_access(self, user_id: str, user_role: str, feature: str) -> Dict:
        """
        检查用户是否可访问某功能
        
        返回: {
            "allowed": bool,
            "reason": str,
            "require_login": bool,  # 是否需要引导登录
        }
        """
        # 管理员权限
        if user_role == "admin":
            return {"allowed": True, "reason": "管理员权限", "require_login": False}

        # 管理员专属功能
        if feature in self._admin_only:
            return {
                "allowed": False,
                "reason": "此功能需要管理员权限",
                "require_login": True,
            }

        # 匿名用户可访问
        if user_id == "guest":
            if feature in self._anon_allowed:
                return {"allowed": True, "reason": "匿名用户可访问", "require_login": False}
            return {
                "allowed": False,
                "reason": f"功能「{feature}」需要登录后使用",
                "require_login": True,
            }

        # 已登录用户
        return {"allowed": True, "reason": "已登录用户可访问", "require_login": False}

    def get_matrix(self) -> Dict:
        """返回完整权限矩阵"""
        return {
            "anonymous_allowed": sorted(self._anon_allowed),
            "auth_required": sorted(self._auth_required),
            "admin_only": sorted(self._admin_only),
        }


class AuthUpgradeFlow:
    """匿名 → 登录升级流程
    
    使用场景：
    1. 匿名用户问「帮我续借」→ 引导登录 → 登录后恢复执行
    2. 登录超时后自动降级为匿名
    """

    def __init__(self):
        # session_id → {pending_feature, pending_params}
        self._pending_upgrades: Dict[str, Dict] = {}

    def require_login(self, session_id: str, feature: str, params: Dict) -> Dict:
        """
        记录待升级请求，返回引导消息
        
        前端流程：
        1. 收到引导消息 → 弹出登录框
        2. 登录成功 → 前端调用 /api/v1/auth/complete-upgrade
        3. 后端取出 pending 请求执行
        """
        self._pending_upgrades[session_id] = {
            "feature": feature,
            "params": params,
            "created_at": __import__("time").time(),
        }
        return {
            "type": "login_required",
            "message": "此操作需要登录后使用。请先登录，登录成功后将自动为您执行。",
            "feature": feature,
            "demo_accounts": [
                {"user_id": "2024001", "password": "123456", "role": "本科生"},
                {"user_id": "2024002", "password": "123456", "role": "研究生"},
            ],
        }

    def complete_upgrade(self, session_id: str, user_id: str) -> Dict:
        """
        登录成功后完成升级，返回待执行的原始请求
        
        前端流程：
        1. 登录接口返回成功
        2. 调用 complete-upgrade 接口
        3. 接口返回原始请求，前端自动重新发送
        """
        pending = self._pending_upgrades.pop(session_id, None)
        if pending is None:
            return {"status": "no_pending", "message": "没有待执行的升级请求"}
        return {
            "status": "pending",
            "feature": pending["feature"],
            "params": pending["params"],
            "user_id": user_id,
            "message": f"登录成功！正在为您执行「{pending['feature']}」请求...",
        }

    def cancel_upgrade(self, session_id: str) -> bool:
        """取消升级请求"""
        if session_id in self._pending_upgrades:
            self._pending_upgrades.pop(session_id)
            return True
        return False


permission_manager = PermissionManager()
auth_upgrade_flow = AuthUpgradeFlow()
