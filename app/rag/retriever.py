"""检索器：双路召回（BM25 稀疏 + 稠密向量）融合 + Cross-Encoder 精排

设计要点（面试可讲）：
- 混合检索：BM25 关键词召回（兜住「3楼」「0.1元」「ISBN」等字面强匹配）
  + BGE-M3 稠密向量召回（抓同义改写、语义泛化），两路互补最稳
- 双路融合分两种：
  1. RRF（Reciprocal Rank Fusion, 默认, k=60）：只看排名、不受两路分数尺度污染，
     某一路失效也鲁棒；加权求和必须各自归一化（见附录B）
  2. 加权求和（weighted）：两路分数各自 min-max 归一化后再按权重相加
- 融合后的候选再走 Cross-Encoder 精排（RERANKER_ENABLED=true）提 Top-N 精度
- 纯稠密模式（BM25 不可用）：退化为「向量召回 + 融合打分（0.75×余弦 + 0.25×词项覆盖）」
- Embedding / 向量库 / BM25 均可替换，本类不做任何假设
- recall() 仅做双路融合召回（用于评测纯净召回率），不走 reranker
"""
from typing import Dict, List, Optional, Tuple

from .chunker import Chunk
from .embeddings import BaseEmbedder, tokenize
from .vector_store import VectorStore
from .reranker import BaseReranker, NoopReranker
from .bm25 import BM25Index


class Retriever:
    """RAG 混合检索器（双路召回 + 融合 + 精排）"""

    FUSION_RRF = "rrf"
    FUSION_WEIGHTED = "weighted"

    # 单路稠密兜底时的融合精排权重：语义分为主，字面覆盖为辅
    SEMANTIC_WEIGHT = 0.75
    LEXICAL_WEIGHT = 0.25

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        reranker: BaseReranker = None,
        bm25: Optional[BM25Index] = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        bm25_candidate_k: int = 20,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker or NoopReranker()
        self.bm25 = bm25
        self.fusion = (fusion or "rrf").lower()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.bm25_candidate_k = bm25_candidate_k

    @property
    def use_hybrid(self) -> bool:
        """是否启用混合检索（需要 BM25 一路可用）"""
        return self.bm25 is not None

    # ---------- 对外接口 ----------
    def recall(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """仅做双路融合召回（不做 Cross-Encoder 精排），用于检索质量纯净评测"""
        if self.use_hybrid:
            return self._hybrid_recall(query, top_k, metadata_filter)
        q_vec = self.embedder.embed_query(query)
        return [
            {"chunk": chunk, "score": round(float(score), 4)}
            for chunk, score in self.store.search(
                q_vec, self.embedder.similarity, top_k=top_k, metadata_filter=metadata_filter
            )
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 3,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """返回精排后的检索结果：[{chunk, score, fusion?/cosine?}, ...]"""
        if self.use_hybrid:
            fused = self._hybrid_recall(query, rerank_top_n, metadata_filter)
        else:
            fused = self._dense_retrieve(query, top_k, rerank_top_n, metadata_filter)

        # Cross-Encoder 精排（直接覆盖 score）
        if self.reranker.name != "noop" and fused:
            return self.reranker.rerank(query, fused)[:rerank_top_n]
        return fused

    # ---------- 混合检索：双路召回 + 融合 ----------
    def _hybrid_recall(
        self, query: str, top_n: int, metadata_filter: Optional[Dict[str, str]]
    ) -> List[Dict]:
        """BM25 一路 + 稠密一路分别召回候选池，再按排名融合取 Top-N"""
        pool_size = max(top_n, self.bm25_candidate_k)
        dense_cands = self._dense_search(query, pool_size, metadata_filter)
        bm25_cands = self.bm25.search(query, pool_size, metadata_filter)

        if self.fusion == self.FUSION_RRF:
            return self._rrf_merge(dense_cands, bm25_cands, top_n)
        return self._weighted_merge(dense_cands, bm25_cands, top_n)

    def _rrf_merge(
        self, dense_cands, bm25_cands, top_n: int
    ) -> List[Dict]:
        """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，只看排名不受分数尺度污染"""
        scores: Dict[str, float] = {}
        for rank, (chunk, _) in enumerate(dense_cands):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
        for rank, (chunk, _) in enumerate(bm25_cands):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )

        chunk_map = {c.chunk_id: c for c, _ in dense_cands}
        chunk_map.update({c.chunk_id: c for c, _ in bm25_cands})
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [
            {"chunk": chunk_map[cid], "score": round(s, 4), "fusion": "rrf"}
            for cid, s in ranked
        ]

    def _weighted_merge(
        self, dense_cands, bm25_cands, top_n: int
    ) -> List[Dict]:
        """加权求和：两路分数各自 min-max 归一化后再按权重相加"""
        dense_norm = self._minmax_norm(dense_cands)
        bm25_norm = self._minmax_norm(bm25_cands)
        combined: Dict[str, float] = {}
        for cid, s in dense_norm.items():
            combined[cid] = combined.get(cid, 0.0) + self.dense_weight * s
        for cid, s in bm25_norm.items():
            combined[cid] = combined.get(cid, 0.0) + self.bm25_weight * s

        chunk_map = {c.chunk_id: c for c, _ in dense_cands}
        chunk_map.update({c.chunk_id: c for c, _ in bm25_cands})
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [
            {"chunk": chunk_map[cid], "score": round(s, 4), "fusion": "weighted"}
            for cid, s in ranked
        ]

    @staticmethod
    def _minmax_norm(cands) -> Dict[str, float]:
        """将 [(chunk, score)] 按分数 min-max 归一化到 [0, 1]，全相等时给 1.0"""
        if not cands:
            return {}
        raw = {c.chunk_id: s for c, s in cands}
        lo, hi = min(raw.values()), max(raw.values())
        if hi - lo < 1e-9:
            return {cid: 1.0 for cid in raw}
        return {cid: (s - lo) / (hi - lo) for cid, s in raw.items()}

    # ---------- 单路稠密兜底 ----------
    def _dense_search(
        self, query: str, top_k: int, metadata_filter: Optional[Dict[str, str]]
    ) -> List[Tuple[Chunk, float]]:
        q_vec = self.embedder.embed_query(query)
        return self.store.search(
            q_vec, self.embedder.similarity, top_k=top_k, metadata_filter=metadata_filter
        )

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        rerank_top_n: int,
        metadata_filter: Optional[Dict[str, str]],
    ) -> List[Dict]:
        """纯稠密模式：向量初检 + 融合打分精排（BM25 不可用时的兜底）"""
        candidates = self._dense_search(query, top_k, metadata_filter)
        cand_dicts = [
            {"chunk": chunk, "cosine": round(float(cos), 4), "score": round(float(cos), 4)}
            for chunk, cos in candidates
        ]
        if self.reranker.name != "noop":
            return self.reranker.rerank(query, cand_dicts)[:rerank_top_n]

        # noop 重排：语义相似度 + 词项覆盖融合打分，兜住字面专有名词
        q_terms = set(tokenize(query))
        for d in cand_dicts:
            doc_terms = set(tokenize(d["chunk"].content))
            overlap = len(q_terms & doc_terms) / (len(q_terms) or 1)
            d["score"] = round(
                self.SEMANTIC_WEIGHT * d["cosine"] + self.LEXICAL_WEIGHT * overlap, 4
            )
        cand_dicts.sort(key=lambda x: x["score"], reverse=True)
        return cand_dicts[:rerank_top_n]
