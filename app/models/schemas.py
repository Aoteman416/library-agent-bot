"""Pydantic 数据模型（请求 / 响应 Schema）"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ================= 请求模型 =================

class ChatMessage(BaseModel):
    """对话消息请求（普通 / 流式共用）"""
    session_id: Optional[str] = Field(None, description="会话ID，新建会话可不传")
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    user_id: Optional[str] = Field(None, description="登录用户学号")


class BookSearchRequest(BaseModel):
    """图书检索请求"""
    keyword: str = Field(..., min_length=1, max_length=200, description="书名/作者/ISBN关键词")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(10, ge=1, le=50, description="每页数量")


class LoginRequest(BaseModel):
    """登录请求（演示用）"""
    user_id: str = Field(..., min_length=1, max_length=32)
    password: str = Field(..., min_length=1, max_length=64)


class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    session_id: str = Field(..., min_length=1, max_length=32)
    user_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5, description="满意度评分 1-5")
    comment: Optional[str] = Field(None, max_length=500, description="改进建议")
    message: Optional[str] = Field(None, max_length=500, description="反馈的对话内容")
    is_helpful: Optional[bool] = Field(None, description="是否有帮助")


class RenewRequest(BaseModel):
    """续借请求（confirm=False 首次请求，True 二次确认）"""
    user_id: str
    book_id: str
    confirm: bool = Field(False, description="二次确认标志")


class SeatReserveRequest(BaseModel):
    """座位预约请求"""
    user_id: str
    area: str = Field(..., description="区域名称")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="预约日期")
    start_time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="开始时间")
    end_time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="结束时间")


# ================= 响应模型 =================

class BookInfo(BaseModel):
    """图书信息"""
    book_id: str
    title: str
    author: str
    isbn: str
    location: str
    available: int
    total: int
    status: str = "available"  # available | unavailable | partial


class ChatReply(BaseModel):
    """对话响应"""
    reply_type: str = "text"  # text | card | image | link | confirm | transfer
    content: str = ""
    extra: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, str]]] = None


class MessageRecord(BaseModel):
    """消息记录"""
    role: str  # user | assistant
    content: str


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    user_id: str
    created_at: str
    messages: List[MessageRecord]
