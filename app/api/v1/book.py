"""馆藏检索接口"""
from fastapi import APIRouter, Query

from ...models.schemas import BookSearchRequest
from ...services.book_service import book_service
from ...utils.helpers import ok

router = APIRouter(prefix="/book", tags=["馆藏"])


@router.get("/search")
async def search_book(
    q: str = Query(..., min_length=1, max_length=200, description="书名/作者/ISBN"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
):
    """馆藏检索"""
    result = book_service.search(q, page=page, page_size=page_size)
    return ok(result)


@router.get("/{book_id}")
async def book_detail(book_id: str):
    """图书详情"""
    book = book_service.get_detail(book_id)
    if book is None:
        return ok(None, message="未找到该书")
    return ok(book)
