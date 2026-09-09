"""FAQ 语义缓存：相似问题命中缓存答案，大幅降低延迟与 LLM 成本

解决的问题（对应审查清单 #11）：
- 高频 FAQ 问题语义缓存，相似度超阈值直接返回缓存答案
- 支持过期机制（TTL）和容量上限（LRU）
- 命中率监控：便于评估缓存价值

设计要点（面试可讲）：
- 基于字符 n-gram TF-IDF 的轻量相似度计算（与 RAG 相同算法）
- 缓存 key 为问题文本，缓存 value 为 (answer, sources, timestamp)
- 可独立运行，无外部依赖
"""
import hashlib
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..rag.embeddings import NgramTfidfEmbedder, cosine_similarity


@dataclass
class CacheEntry:
    question: str
    answer: str
    sources: Optional[List[Dict]]
    vector: Dict[str, float]
    created_at: float
    hit_count: int = 0


class SemanticCache:
    """语义缓存：相似问题命中相同答案"""

    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.85, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embedder = NgramTfidfEmbedder()
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._hit_count = 0
        self._miss_count = 0

    def lookup(self, question: str) -> Optional[Tuple[str, Optional[List[Dict]]]]:
        """
        查找缓存：精确匹配 → 语义匹配
        
        返回: (answer, sources) 或 None
        """
        # 先做精确匹配
        question_lower = question.lower().strip()
        entry = self._cache.get(question_lower)
        if entry:
            self._touch(entry)
            self._hit_count += 1
            return (entry.answer, entry.sources)

        # 语义匹配：计算与所有缓存项的向量相似度
        if self._cache:
            q_vec = self._embedder.embed(question)
            best_score = 0.0
            best_entry = None

            for entry in self._cache.values():
                score = cosine_similarity(q_vec, entry.vector)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry and best_score >= self._threshold:
                self._touch(best_entry)
                best_entry.hit_count += 1
                self._hit_count += 1
                return (best_entry.answer, best_entry.sources)

        self._miss_count += 1
        return None

    def store(self, question: str, answer: str, sources: Optional[List[Dict]] = None) -> None:
        """存入缓存"""
        question_lower = question.lower().strip()
        # 若已存在则更新
        if question_lower in self._cache:
            self._cache.move_to_end(question_lower)

        self._cache[question_lower] = CacheEntry(
            question=question,
            answer=answer,
            sources=sources,
            vector=self._embedder.embed(question),
            created_at=time.time(),
        )

        # LRU 淘汰
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def _touch(self, entry: CacheEntry) -> None:
        """更新 LRU 位置 + 过期检查"""
        # 过期检查
        if time.time() - entry.created_at > self._ttl:
            self._cache.pop(entry.question.lower())
            return
        self._cache.move_to_end(entry.question.lower())

    def invalidate(self, question: str) -> bool:
        """删除缓存项"""
        key = question.lower().strip()
        if key in self._cache:
            self._cache.pop(key)
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> Dict:
        total = self._hit_count + self._miss_count or 1
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total * 100, 1),
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
        }


class RedisSemanticCache:
    """基于 Redis 的语义缓存后端（对应 v2 9.1 缓存）

    与进程内 SemanticCache 接口一致；Redis 不可用时自动回退本地内存，
    保证链路不中断（与全局降级策略一致）。
    """

    def __init__(
        self,
        client,
        max_size: int = 500,
        similarity_threshold: float = 0.85,
        ttl_seconds: int = 3600,
        embedder: NgramTfidfEmbedder = None,
    ):
        self._client = client
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._embedder = embedder or NgramTfidfEmbedder()
        self._hit_count = 0
        self._miss_count = 0
        # 本地兜底（Redis 抖动时）
        self._local: "OrderedDict[str, CacheEntry]" = OrderedDict()

    def _key(self, question: str) -> str:
        return "faq_cache:" + hashlib.md5(question.lower().strip().encode()).hexdigest()

    def lookup(self, question: str) -> Optional[Tuple[str, Optional[List[Dict]]]]:
        q = question.lower().strip()
        key = self._key(q)
        try:
            raw = self._client.get(key)
            if raw:
                entry: CacheEntry = pickle.loads(raw)
                self._hit_count += 1
                return (entry.answer, entry.sources)
        except Exception:  # noqa: BLE001
            pass
        # 回退本地
        entry = self._local.get(q)
        if entry:
            self._hit_count += 1
            return (entry.answer, entry.sources)
        self._miss_count += 1
        return None

    def store(self, question: str, answer: str, sources: Optional[List[Dict]] = None) -> None:
        q = question.lower().strip()
        entry = CacheEntry(
            question=question,
            answer=answer,
            sources=sources,
            vector=self._embedder.embed(question),
            created_at=time.time(),
        )
        try:
            self._client.set(self._key(q), pickle.dumps(entry), ex=self._ttl)
        except Exception:  # noqa: BLE001
            self._local[q] = entry
            while len(self._local) > self._max_size:
                self._local.popitem(last=False)

    def invalidate(self, question: str) -> bool:
        key = self._key(question.lower().strip())
        try:
            self._client.delete(key)
        except Exception:  # noqa: BLE001
            pass
        return self._local.pop(question.lower().strip(), None) is not None

    def clear(self) -> None:
        try:
            for k in self._client.scan_iter(match="faq_cache:*"):
                self._client.delete(k)
        except Exception:  # noqa: BLE001
            pass
        self._local.clear()

    def stats(self) -> Dict:
        total = self._hit_count + self._miss_count or 1
        return {
            "backend": "redis",
            "size": "n/a",
            "max_size": self._max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total * 100, 1),
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
        }


def create_cache():
    """工厂：按配置选择缓存后端（Redis / 进程内）"""
    from ..config import settings
    from .logger import logger

    if settings.REDIS_ENABLED:
        try:
            import redis as redis_lib
            client = redis_lib.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD or None,
                socket_connect_timeout=1,
            )
            client.ping()
            logger.info(
                "FAQ 语义缓存使用 Redis 后端（%s:%s db=%s）",
                settings.REDIS_HOST, settings.REDIS_PORT, settings.REDIS_DB,
            )
            return RedisSemanticCache(
                client,
                max_size=settings.REDIS_MAX_SIZE or 500,
                ttl_seconds=settings.REDIS_TTL,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 不可用，FAQ 缓存回退进程内：%s", exc)
    return SemanticCache(
        max_size=settings.REDIS_MAX_SIZE or 500,
        similarity_threshold=0.85,
        ttl_seconds=settings.REDIS_TTL,
    )


# 全局单例（按配置选择后端）
faq_cache = create_cache()
