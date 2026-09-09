"""通用工具函数"""
from typing import Any, Dict, List, Optional


def ok(data: Any = None, message: str = "success") -> Dict[str, Any]:
    """统一成功响应"""
    return {"code": 0, "message": message, "data": data}


def fail(code: int, message: str, data: Any = None) -> Dict[str, Any]:
    """统一失败响应"""
    return {"code": code, "message": message, "data": data}


def sse_event(event: str, data: Any) -> str:
    """构造 SSE 事件报文（data 以 JSON 序列化）"""
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, size: int = 8) -> List[str]:
    """将文本切成小片段，用于流式输出演示"""
    if not text:
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]
