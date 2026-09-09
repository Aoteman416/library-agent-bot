"""向量存储：内存稀疏索引（默认） / Chroma 持久化稠密索引（可选）

设计要点（面试可讲）：
- 两种实现暴露同一组接口：add / search / size / clear / get_chunk，
  RAG 上层（Retriever）无需改动即可切换
- 内存实现：零依赖，进程内线性扫描，适合 <10 万条的小知识库与 CI 环境
- Chroma 实现：PersistentClient 落盘 + HNSW 近似最近邻 + 原生元数据过滤，
  进程重启后直接复用索引，不必重跑 Embedding
- 接入点集中在 create_store 工厂：切换只需改 VECTOR_STORE_MODE 一个配置
"""
import json
from typing import Dict, List, Optional, Tuple

from .chunker import Chunk
from .embeddings import Vector, _as_list


class VectorStore:
    """基于稀疏向量的内存向量库（零依赖，默认实现）"""

    name = "memory"

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._vectors: List[Vector] = []
        self._index_by_id: Dict[str, int] = {}

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._index_by_id.clear()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk, vector: Vector) -> None:
        self._index_by_id[chunk.chunk_id] = len(self._chunks)
        self._chunks.append(chunk)
        self._vectors.append(vector)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._index_by_id.get(chunk_id)
        return self._chunks[idx] if idx is not None else None

    def search(
        self,
        query_vec: Vector,
        similarity,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """检索 Top-K：支持可选的元数据过滤"""
        scored: List[Tuple[Chunk, float]] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            if metadata_filter and not self._match_meta(chunk, metadata_filter):
                continue
            score = similarity(query_vec, vec)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _match_meta(chunk: Chunk, metadata_filter: Dict[str, str]) -> bool:
        return all(chunk.metadata.get(k) == v for k, v in metadata_filter.items())

    # ---------- 持久化 ----------
    def to_json(self) -> Dict:
        return {
            "chunks": [
                {"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
                for c in self._chunks
            ],
            "vectors": self._vectors,
        }

    def load_json(self, data: Dict) -> None:
        self.clear()
        chunks = data.get("chunks", [])
        vectors = data.get("vectors", [])
        for c, v in zip(chunks, vectors):
            self.add(
                Chunk(
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    metadata=c.get("metadata", {}),
                ),
                v,
            )


class ChromaStore:
    """Chroma 持久化向量库（稠密向量 + HNSW 检索）

    与 VectorStore 保持接口一致，差异仅在内部实现：
    - 数据落盘到 CHROMA_PERSIST_DIR，进程重启后 size 不变即可直接复用
    - 使用 cosine 空间，Chroma 返回 cosine distance，需转换回相似度分数
    - 元数据过滤走 Chroma 原生 where 语法（多条件为隐式 AND）
    """

    name = "chroma"

    def __init__(
        self,
        persist_dir: str = "",
        collection_name: str = "",
        distance: str = "cosine",
    ):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        from ..config import settings

        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": distance},
        )

    @property
    def size(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """清空集合（Chroma 无 truncate，采用删除后重建）"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunk: Chunk, vector: Vector) -> None:
        self._collection.add(
            ids=[chunk.chunk_id],
            embeddings=[_as_list(vector)],
            documents=[chunk.content],
            # Chroma metadata 仅支持 str/int/float/bool，统一转字符串
            metadatas=[{k: str(v) for k, v in chunk.metadata.items()}],
        )

    def add_batch(self, chunks: List[Chunk], vectors: List[Vector]) -> None:
        """批量写入：减少与 Chroma 的往返次数"""
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[_as_list(v) for v in vectors],
            documents=[c.content for c in chunks],
            metadatas=[{k: str(v) for k, v in c.metadata.items()} for c in chunks],
        )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        result = self._collection.get(ids=[chunk_id])
        if not result["ids"]:
            return None
        return Chunk(
            chunk_id=result["ids"][0],
            content=result["documents"][0] or "",
            metadata=dict(result["metadatas"][0] or {}),
        )

    def search(
        self,
        query_vec: Vector,
        similarity=None,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """检索 Top-K：由 Chroma 内部完成 ANN 检索，similarity 参数被忽略"""
        total = self.size
        if total == 0:
            return []

        kwargs = {
            "query_embeddings": [_as_list(query_vec)],
            "n_results": max(1, min(top_k, total)),
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = {k: {"$eq": v} for k, v in metadata_filter.items()}

        result = self._collection.query(**kwargs)
        rows = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
        return [
            (
                Chunk(chunk_id=cid, content=doc or "", metadata=dict(meta or {})),
                self._distance_to_score(dist),
            )
            for cid, doc, meta, dist in rows
        ]

    @staticmethod
    def _distance_to_score(distance) -> float:
        """Chroma 在 cosine 空间返回 cosine distance（1 - 余弦相似度）"""
        return round(1.0 - float(distance), 4)

    def to_json(self) -> Dict:
        """Chroma 自身已持久化，此处仅导出元数据用于索引一致性检查"""
        dumped = self._collection.get(include=["metadatas"])
        return {"chunks": dumped["ids"], "vectors": dumped["metadatas"]}

    def load_json(self, data: Dict) -> None:  # pragma: no cover - 由 Chroma 落盘负责
        raise NotImplementedError("Chroma 由 PersistentClient 自动落盘，无需手动 load")


def create_store(mode: str = "memory", **kwargs):
    """工厂方法：按配置创建向量库

    memory → 内存稀疏索引（默认，零依赖）
    chroma → Chroma 持久化索引；依赖缺失或初始化失败时自动回退到 memory
    """
    from ..core.logger import logger

    if (mode or "memory").strip().lower() == "chroma":
        try:
            return ChromaStore(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 初始化失败，回退到内存向量库：%s", exc)
    return VectorStore()
