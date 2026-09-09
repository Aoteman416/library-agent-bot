"""Embedding 模块：稀疏（字符 n-gram + TF-IDF）与稠密（BGE / sentence-transformers）双实现

设计要点（面试可讲）：
- 统一接口：fit / embed / embed_batch / embed_query / similarity，
  RAG 上层（Retriever、VectorStore）完全不感知底层是稀疏还是稠密向量
- 稀疏实现：中文不做分词，用「单字 + 相邻双字 n-gram」作特征项 + TF-IDF 加权，
  零外部依赖、冷启动毫秒级；适合小知识库、离线演示与 CI 环境
- 稠密实现：BAAI/bge-large-zh-v1.5（1024 维），中文语义匹配强，可召回同义改写；
  查询侧追加 bge 官方检索指令前缀，与文档侧向量保持同一语义空间
- 两种实现通过 EMBEDDING_MODE 一键切换，构成 A/B 对比评测的实验变量
"""
import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Union

# 常见无信息量字词，降低检索噪声
_STOPWORDS = frozenset(
    "的了在和有就是都而及与或一个是我你他她它我们你们它们这那某"
    "请问怎么如何可以需要帮助想要哪个哪些什么哪层楼地方里的和为"
    "没有吗呢吧啊呀哦嗯喂你好老师同学工作人员图书馆"
)

_NON_WORD = re.compile(r"[^\u4e00-\u9fff0-9a-zA-Z]")

# 向量类型：稀疏为 {term: weight}，稠密为 [float, ...]
Vector = Union[Dict[str, float], List[float]]


def tokenize(text: str) -> List[str]:
    """字符 n-gram 特征项（单字 + 相邻双字），过滤停用词"""
    text = _NON_WORD.sub("", text.lower())
    if not text:
        return []
    grams: List[str] = []
    for i, ch in enumerate(text):
        if ch not in _STOPWORDS:
            grams.append(ch)
        if i + 1 < len(text):
            bigram = text[i:i + 2]
            if bigram not in _STOPWORDS:
                grams.append(bigram)
    return grams


def _as_list(vec) -> List[float]:
    """numpy 数组 / torch 张量 / 列表 → 统一为 float 列表"""
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def _dense_cosine(vec_a, vec_b) -> float:
    """稠密向量余弦相似度"""
    a = _as_list(vec_a)
    b = _as_list(vec_b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _sparse_cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """稀疏向量余弦相似度（只在共有词项上计算）"""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """余弦相似度：自动适配稀疏字典向量与稠密列表向量"""
    if isinstance(vec_a, dict) or isinstance(vec_b, dict):
        return _sparse_cosine(vec_a, vec_b)
    return _dense_cosine(vec_a, vec_b)


class BaseEmbedder:
    """Embedding 统一接口

    子类只需实现 embed()，其余方法提供合理默认实现，
    保证 Retriever / VectorStore 对稀疏与稠密实现无感知。
    """

    name: str = "base"
    dimension: int = 0

    def fit(self, corpus: Sequence[str]) -> "BaseEmbedder":
        """预训练钩子：稀疏实现用它算 IDF，稠密实现为空操作"""
        return self

    def embed(self, text: str) -> Vector:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> List[Vector]:
        """批量编码：稠密模型会真正走 batch 推理，稀疏实现退化为逐条"""
        return [self.embed(t) for t in texts]

    def embed_query(self, query: str) -> Vector:
        """检索侧编码（稠密模型通常需要 query 指令前缀）"""
        return self.embed(query)

    def similarity(self, vec_a: Vector, vec_b: Vector) -> float:
        return cosine_similarity(vec_a, vec_b)


class NgramTfidfEmbedder(BaseEmbedder):
    """字符 n-gram + TF-IDF 中文检索嵌入器（零依赖）"""

    name = "ngram-tfidf"

    def __init__(self, n: int = 2):
        self.n = n
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def fit(self, corpus: Sequence[str]) -> "NgramTfidfEmbedder":
        """基于语料计算 IDF 权重"""
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(tokenize(doc)))
        self._doc_count = max(len(corpus), 1)
        self._idf = {
            term: math.log((self._doc_count + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }
        return self

    def embed(self, text: str) -> Dict[str, float]:
        """文本 → 稀疏 TF-IDF 向量"""
        tf = Counter(tokenize(text))
        total = sum(tf.values()) or 1
        return {
            term: (freq / total) * self._idf.get(term, 1.0)
            for term, freq in tf.items()
        }

    def similarity(self, vec_a: Vector, vec_b: Vector) -> float:
        return cosine_similarity(vec_a, vec_b)


class SentenceTransformerEmbedder(BaseEmbedder):
    """稠密 Embedding：BGE / sentence-transformers 系列模型

    默认 BAAI/bge-large-zh-v1.5（1024 维，中文检索 SOTA 级别，模型约 1.3GB）。
    机器资源有限时可改用 BAAI/bge-small-zh-v1.5（512 维，约 100MB）。

    查询侧前缀是 bge 系列的官方建议：短查询加指令前缀后与长文档处于同一语义空间，
    实测对「问句式 query 匹配陈述式文档」的召回率有明显提升。
    """

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "",
        device: str = "",
        batch_size: int = 0,
        query_prefix: str = "",
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底
            raise ImportError(
                "未安装 sentence-transformers，无法使用稠密 Embedding。\n"
                "安装：pip install sentence-transformers torch\n"
                "或设置 EMBEDDING_MODE=ngram 使用零依赖的稀疏检索。"
            ) from exc

        from ..config import settings

        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.query_prefix = (
            settings.EMBEDDING_QUERY_PREFIX if query_prefix == "" else query_prefix
        )
        self.normalize = normalize

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode(list(texts))

    def embed_query(self, query: str) -> List[float]:
        if self.query_prefix:
            query = f"{self.query_prefix}{query}"
        return self.embed(query)

    def similarity(self, vec_a: Vector, vec_b: Vector) -> float:
        return cosine_similarity(vec_a, vec_b)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [_as_list(v) for v in vectors]


def create_embedder(mode: str = "ngram", **kwargs) -> BaseEmbedder:
    """工厂方法：按配置创建嵌入器

    ngram    → 零依赖稀疏 TF-IDF（默认，开箱即用）
    external → 稠密 BGE 模型；依赖缺失或初始化失败时自动回退到 ngram，
               保证服务始终可用（与服务降级策略一致）
    """
    from ..core.logger import logger

    mode = (mode or "ngram").strip().lower()
    if mode in ("external", "dense", "bge", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("稠密 Embedding 初始化失败，回退到稀疏 n-gram 模式：%s", exc)
    return NgramTfidfEmbedder()
