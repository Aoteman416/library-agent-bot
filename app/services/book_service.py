"""馆藏检索服务

当前为 Mock 数据演示；对接真实 ILS（图书馆自动化系统）时，
将各方法内部替换为对 OPAC / ILS API 的 HTTP 调用即可。
"""
from typing import Dict, List, Optional

from ..models.schemas import BookInfo

# 模拟馆藏数据（演示用）
MOCK_BOOKS = [
    {
        "book_id": "B001",
        "title": "三体",
        "author": "刘慈欣",
        "isbn": "978-7-5366-9293-0",
        "location": "3楼A区23架",
        "available": 2,
        "total": 5,
    },
    {
        "book_id": "B002",
        "title": "C++ Primer 第六版",
        "author": "Lippman",
        "isbn": "978-7-121-15526-2",
        "location": "4楼B区07架",
        "available": 1,
        "total": 3,
    },
    {
        "book_id": "B003",
        "title": "Python编程：从入门到实践",
        "author": "Eric Matthes",
        "isbn": "978-7-115-42802-8",
        "location": "4楼B区12架",
        "available": 3,
        "total": 4,
    },
    {
        "book_id": "B004",
        "title": "深度学习",
        "author": "Ian Goodfellow",
        "isbn": "978-7-115-46147-6",
        "location": "4楼B区15架",
        "available": 0,
        "total": 2,
    },
    {
        "book_id": "B005",
        "title": "机器学习",
        "author": "周志华",
        "isbn": "978-7-302-40263-7",
        "location": "4楼B区16架",
        "available": 4,
        "total": 6,
    },
]


def _status(book: Dict) -> str:
    if book["available"] == 0:
        return "unavailable"
    if book["available"] < book["total"]:
        return "partial"
    return "available"


# 检索关键词中的无意义填充词，抽取真正的书名/作者/ISBN
_FILLERS = [
    "图书馆", "有没有在架", "有没有这", "有没有", "在架吗", "在架", "在不在",
    "在哪", "在哪里", "哪里", "借阅状态", "可借", "能借", "怎么", "如何",
    "这本书", "那本书", "那本", "这本书吗", "那本书吗", "这本", "的书",
    "书吗", "是不", "是不是", "我想", "我要", "想", "帮我", "借一本", "查一下",
    "找一下", "看看", "吗", "呢", "？", "?", "请问", "《", "》", "有",
]


def _strip_quotes(text: str) -> str:
    return text.replace("《", "").replace("》", "")


def _clean_keyword(keyword: str) -> str:
    """从自然语句中抽取图书检索关键词"""
    # 优先取书名号内容
    import re as _re
    m = _re.search(r"《([^》]+)》", keyword)
    if m:
        return m.group(1).strip()
    kw = keyword.strip()
    # 去掉语气/查询填充词，从长到短替换，避免误删有效词
    for filler in sorted(_FILLERS, key=len, reverse=True):
        kw = kw.replace(filler, "")
    kw = kw.strip()
    if kw.endswith("书") or kw.endswith(("吗", "呢")):
        pass
    return kw or _strip_quotes(keyword.strip())


class BookService:
    """馆藏检索服务"""

    def search(self, keyword: str, page: int = 1, page_size: int = 10) -> Dict:
        kw = _clean_keyword(keyword).lower()
        if not kw:
            return {"total": 0, "books": []}

        matched = [
            b
            for b in MOCK_BOOKS
            if kw in b["title"].lower()
            or kw in b["author"].lower()
            or kw in b["isbn"]
        ]

        start = (page - 1) * page_size
        page_books = matched[start:start + page_size]
        books = [
            BookInfo(**{**b, "status": _status(b)}).model_dump() for b in page_books
        ]
        return {"total": len(matched), "books": books}

    def get_detail(self, book_id: str) -> Optional[Dict]:
        for b in MOCK_BOOKS:
            if b["book_id"] == book_id:
                return {**b, "status": _status(b)}
        return None


book_service = BookService()
