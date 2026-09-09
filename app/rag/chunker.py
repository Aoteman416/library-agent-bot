"""文本分块器（自适应分块，对应需求分析 v2 3.2.1）

设计要点（面试可讲）：
- 中文场景按「段落 → 句子 → 字符窗口」三级递归切分
- 自适应策略：按文档类型（doc_type）选择分块大小与重叠
  - policy_doc 政策文件：chunk_size=500, overlap=100
  - faq_doc    FAQ 问答：chunk_size=300, overlap=0
  - guide_doc  使用指南：chunk_size=800, overlap=150
  - 未分类（general）：沿用全局 CHUNK_SIZE / CHUNK_OVERLAP
- 通过 chunk_overlap 保留相邻块上下文衔接
- 分块携带元数据（来源 / 分类 / 类型），供检索时过滤
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import settings


_SENT_END = re.compile(r"(?<=[。！？；?!])")
_PARAGRAPH_SPLIT = re.compile(r"\n{1,}")


@dataclass
class Chunk:
    """知识库分块"""
    chunk_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


class TextChunker:
    """自适应文本分块器"""

    DEFAULT_STRATEGY = {"chunk_size": 200, "chunk_overlap": 50}

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategies: Dict = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # 自适应分块策略（policy / faq / guide），来自配置
        self.strategies = dict(strategies) if strategies is not None else dict(settings.CHUNK_STRATEGY)

    def _resolve(self, metadata: Dict[str, str]) -> tuple:
        """根据文档类型解析 (chunk_size, chunk_overlap)，并做安全 clamp"""
        doc_type = (metadata or {}).get("doc_type", "")
        st = self.strategies.get(doc_type, {})
        size = int(st.get("chunk_size", self.chunk_size))
        overlap = int(st.get("chunk_overlap", self.chunk_overlap))
        if overlap >= size:
            overlap = max(0, size - 1)
        return size, overlap

    def split_text(self, text: str, metadata: Dict[str, str]) -> List[Chunk]:
        """将整篇文档切分为 Chunk 列表（按 doc_type 自适应分块）"""
        size, overlap = self._resolve(metadata)
        text = text.strip()
        if not text:
            return []

        sentences: List[str] = []
        for para in _PARAGRAPH_SPLIT.split(text):
            para = para.strip()
            if not para:
                continue
            for sent in _SENT_END.split(para):
                sent = sent.strip()
                if sent:
                    sentences.append(sent)

        chunks: List[Chunk] = []
        buffer = ""
        for sent in sentences:
            if len(sent) > size:
                if buffer:
                    chunks.append(self._make_chunk(buffer, metadata))
                    buffer = ""
                chunks.extend(self._hard_split(sent, metadata, size, overlap))
                continue

            if len(buffer) + len(sent) + 1 <= size or not buffer:
                buffer = f"{buffer}{sent}" if buffer else sent
            else:
                chunks.append(self._make_chunk(buffer, metadata))
                buffer = buffer[-overlap:] + sent if overlap else sent

        if buffer:
            chunks.append(self._make_chunk(buffer, metadata))

        return chunks

    def _make_chunk(self, content: str, metadata: Dict[str, str]) -> Chunk:
        return Chunk(
            chunk_id=uuid.uuid4().hex[:12],
            content=content,
            metadata=dict(metadata),
        )

    def _hard_split(self, text: str, metadata: Dict[str, str], size: int, overlap: int) -> List[Chunk]:
        """超长文本按固定窗口硬切（带重叠）"""
        out = []
        if overlap >= size:
            overlap = max(0, size - 1)
        step = size - overlap
        if step <= 0:
            step = size
        for i in range(0, len(text), step):
            out.append(self._make_chunk(text[i:i + size], metadata))
        return out
