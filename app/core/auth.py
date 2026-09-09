"""JWT 鉴权（对应需求分析 v2 第九节 9.1 认证）

设计要点：
- Access Token（短期，默认 60 分钟）+ Refresh Token（长期，默认 7 天）
- 优先使用 python-jose（依赖可选）；未安装时自动降级为内置 hmac-sha256 实现，
  保证服务在缺依赖环境下仍可签发/校验 token（与全局降级策略一致）
- get_current_user：从 Authorization: Bearer <token> 解析；
  兼容旧版 X-Demo-User / X-Demo-Role 头（演示用），保证现有前端/接口不破坏
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header

from ..config import settings
from .errors import UnauthorizedError
from .logger import logger

try:
    from jose import jwt as _jose_jwt, JWTError
    _HAS_JOSE = True
except ImportError:  # pragma: no cover - 缺依赖时降级
    _jose_jwt = None
    JWTError = Exception
    _HAS_JOSE = False


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def create_access_token(user_id: str, role: str = "student") -> str:
    """签发 Access Token（短期）"""
    return _encode(
        {"sub": user_id, "role": role, "type": "access"},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str, role: str = "student") -> str:
    """签发 Refresh Token（长期）"""
    return _encode(
        {"sub": user_id, "role": role, "type": "refresh"},
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def _encode(payload: dict, expires_delta: timedelta) -> str:
    payload = dict(payload)
    payload["iat"] = _now_ts()
    payload["exp"] = _now_ts() + int(expires_delta.total_seconds())
    if _HAS_JOSE:
        return _jose_jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return _hmac_encode(payload)


def _b64url_encode(raw: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _hmac_encode(payload: dict) -> str:
    import base64, hashlib, hmac, json
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, ensure_ascii=False).encode())
    body = _b64url_encode(json.dumps(payload, ensure_ascii=False).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(settings.JWT_SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"


def decode_token(token: str) -> dict:
    """校验并解析 token，失败抛 UnauthorizedError"""
    if _HAS_JOSE:
        try:
            return _jose_jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        except JWTError as exc:
            raise UnauthorizedError(str(exc))
    return _hmac_decode(token)


def _hmac_decode(token: str) -> dict:
    import base64, hashlib, hmac, json
    try:
        header, body, sig = token.split(".")
    except ValueError:
        raise UnauthorizedError("token 格式错误")
    signing_input = f"{header}.{body}".encode()
    expected = hmac.new(settings.JWT_SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    try:
        sig_bytes = _b64url_decode(sig)
    except Exception:
        raise UnauthorizedError("token 签名错误")
    if not hmac.compare_digest(expected, sig_bytes):
        raise UnauthorizedError("token 签名校验失败")
    try:
        payload = json.loads(_b64url_decode(body).decode())
    except Exception:
        raise UnauthorizedError("token 解析失败")
    if payload.get("exp", 0) < _now_ts():
        raise UnauthorizedError("token 已过期")
    return payload


class CurrentUser:
    """当前登录用户"""

    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role

    @property
    def is_guest(self) -> bool:
        return self.user_id == "guest"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_demo_user: Optional[str] = Header(default=None),
    x_demo_role: Optional[str] = Header(default=None),
) -> CurrentUser:
    """解析当前用户：优先 Bearer Token，其次 X-Demo-User（演示兼容）"""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise UnauthorizedError("请使用 access token 调用业务接口")
        return CurrentUser(payload.get("sub", "guest"), payload.get("role", "student"))
    if x_demo_user:
        return CurrentUser(x_demo_user, x_demo_role or "student")
    # 默认匿名
    return CurrentUser("guest", "guest")
