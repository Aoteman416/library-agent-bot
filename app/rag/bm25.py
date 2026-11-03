"""BM25 索引：中文关键词召回（混合检索的稀疏一路）

设计要点（面试可讲）：
- 用 rank_bm25 的 BM25Okapi 做关键词召回；中文用 jieba 分词（缺失时回退到字/2-gram），
  解决 rank_bm25 默认按空格分词对中文直接失效的坑
- 与 VectorStore 接口对齐：add_batch / search 返回 [(chunk, score)]，Retriever 不感知
  底层是向量还是倒排，双路索引可独立替换
- 仅在 HYBRID_RETRIEVAL / BM25_ENABLED 开启时构建；rank_bm25 依赖缺失自动降级（bm25=None），
  服务不中断（与全局降级策略一致）
- BM25 是内存倒排索引，随知识库更新需重建（热加载/独立服务是生产演进方向，见附录B）
"""
from typing import Dict, List, Optional, Tuple

from .chunker import Chunk


def _default_tokenize(text: str) -> List[str]:
    """中文分词：优先 jieba，缺失时回退到字符 n-gram（单字 + 相邻双字）"""
    try:
        import jieba

        return [t for t in jieba.lcut(text) if t.strip()]
    except Exception:  # pragma: no cover - 无 jieba 时的兜底
        import re

        text = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text.lower())
        grams: List[str] = []
        for i, ch in enumerate(text):
            grams.append(ch)
            if i + 1 < len(text):
                grams.append(text[i : i + 2])
        return grams


class BM25Index:
    """基于倒排索引的 BM25 关键词召回（中文友好）"""

    name = "bm25"

    def __init__(
        self,
        tokenizer=None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._tokenizer = tokenizer or _default_tokenize
        self._k1 = k1
        self._b = b
        self._chunks: List[Chunk] = []
        self._texts: List[str] = []
        self._index_by_id: Dict[str, int] = {}
        self._bm25 = None  # 懒构建，add 后首次 search 时建

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk, text: str) -> None:
        self._append(chunk, text)

    def add_batch(self, chunks: List[Chunk], texts: List[str]) -> None:
        for chunk, text in zip(chunks, texts):
            self._append(chunk, text)

    def _append(self, chunk: Chunk, text: str) -> None:
        self._index_by_id[chunk.chunk_id] = len(self._chunks)
        self._chunks.append(chunk)
        self._texts.append(text)
        self._bm25 = None  # 标记脏，下次 search 重建

    def _ensure_built(self) -> None:
        if self._bm25 is None and self._chunks:
            from rank_bm25 import BM25Okapi

            corpus = [self._tokenizer(t) for t in self._texts]
            self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """关键词召回 Top-K：支持可选的元数据过滤"""
        self._ensure_built()
        if not self._bm25:
            return []
        q_tokens = self._tokenizer(query)
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

        results: List[Tuple[Chunk, float]] = []
        for i in ranked:
            if float(scores[i]) <= 0:
                continue
            chunk = self._chunks[i]
            if metadata_filter and not all(
                chunk.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            results.append((chunk, round(float(scores[i]), 4)))
            if len(results) >= top_k:
                break
        return results

    def clear(self) -> None:
        self._chunks.clear()
        self._texts.clear()
        self._index_by_id.clear()
        self._bm25 = None
