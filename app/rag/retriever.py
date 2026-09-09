"""检索器：向量初检 + 融合精排 / Cross-Encoder 精排

设计要点（面试可讲）：
- 两阶段检索：向量相似度初检（Top-K）→ 精排（Top-N）
- 精排分两种模式：
  1. 无重排器（默认，零依赖）：语义相似度 0.75 + 词项覆盖 0.25 的融合打分，
     叠加字面词项覆盖兜住「3楼」「0.1元」等专有名词的精确匹配
  2. Cross-Encoder 重排器（RERANKER_ENABLED=true）：对「查询-文档」对做精细相关性打分，
     显著提升 Top-N 精度（对应 v2 3.1 Step3）
- Embedding 与向量库均可替换（稀疏/稠密、内存/Chroma），本类不做任何假设
- recall() 仅做向量召回（用于评测纯净召回率），不走 reranker
"""
from typing import Dict, List, Optional

from .chunker import Chunk
from .embeddings import BaseEmbedder, tokenize
from .vector_store import VectorStore
from .reranker import BaseReranker, NoopReranker


class Retriever:
    """RAG 检索器（召回 + 精排）"""

    # 融合精排权重：语义分为主，字面覆盖为辅
    SEMANTIC_WEIGHT = 0.75
    LEXICAL_WEIGHT = 0.25

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        reranker: BaseReranker = None,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker or NoopReranker()

    def recall(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """仅做向量召回（不做精排），用于检索质量的纯净评测"""
        q_vec = self.embedder.embed_query(query)
        return [
            {"chunk": chunk, "score": round(float(score), 4)}
            for chunk, score in self.store.search(
                q_vec,
                self.embedder.similarity,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 3,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """返回精排后的检索结果：[{chunk, score, cosine}, ...]"""
        q_vec = self.embedder.embed_query(query)

        # 阶段一：向量初检
        candidates = self.store.search(
            q_vec,
            self.embedder.similarity,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        cand_dicts = [
            {"chunk": chunk, "cosine": round(float(cos), 4), "score": round(float(cos), 4)}
            for chunk, cos in candidates
        ]

        # 阶段二：精排
        if self.reranker.name == "noop":
            # 融合打分：语义相似度 + 词项覆盖
            q_terms = set(tokenize(query))
            for d in cand_dicts:
                doc_terms = set(tokenize(d["chunk"].content))
                overlap = len(q_terms & doc_terms) / (len(q_terms) or 1)
                d["score"] = round(
                    self.SEMANTIC_WEIGHT * d["cosine"] + self.LEXICAL_WEIGHT * overlap, 4
                )
            cand_dicts.sort(key=lambda x: x["score"], reverse=True)
            return cand_dicts[:rerank_top_n]

        # Cross-Encoder 重排（直接覆盖 score）
        reranked = self.reranker.rerank(query, cand_dicts)
        return reranked[:rerank_top_n]
