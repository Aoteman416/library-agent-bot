"""知识库服务：加载 data/knowledge 下的 Markdown 文档

设计要点（面试可讲）：
- 支持 .md / .txt 文档的加载
- 将文档标题（一级标题）作为分类元数据，供检索过滤
- 按文件名/章节标题推断 doc_type（policy/faq/guide/general），驱动自适应分块（见 chunker.py）
- 上传新文档后调用 reindex 即可增量更新检索索引
"""
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings
from ..core.logger import logger
from ..rag.chunker import Chunk, TextChunker


def _parse_md(text: str) -> List[Dict[str, str]]:
    """解析 Markdown：返回 [{title, content}]，按一级/二级标题分节"""
    sections: List[Dict[str, str]] = []
    current_title = "综合"
    current_lines: List[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"title": current_title, "content": body})

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            flush()
            current_lines = []
            current_title = stripped.lstrip("# ").strip()
        elif stripped.startswith("## "):
            flush()
            current_lines = []
            current_title = stripped.lstrip("# ").strip()
        elif stripped.startswith("#"):
            continue  # 忽略更深层级标题文本
        else:
            current_lines.append(line)
    flush()
    return sections


class KnowledgeService:
    """知识库服务：文档 → 分块 → 元数据"""

    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or settings.KNOWLEDGE_DIR
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self._chunks: List[Chunk] = []
        self._documents: List[Dict] = []

    # ---------- 文档类型推断 ----------
    def _infer_doc_type(self, file_name: str, title: str) -> str:
        """推断文档类型（doc_type），用于自适应分块策略选择

        优先级：文件名关键字 > 章节标题关键字 > general
        """
        text = f"{file_name} {title}".lower()
        if any(k in text for k in ("faq", "问答", "常见问题", "q&a")):
            return "faq_doc"
        if any(k in text for k in ("指南", "guide", "使用", "教程", "手册", "操作")):
            return "guide_doc"
        if any(k in text for k in ("规则", "政策", "规定", "办法", "条例", "开放时间", "制度")):
            return "policy_doc"
        return "general"

    # ---------- 文档加载 ----------
    def load_directory(self) -> int:
        """加载知识库目录下全部文档，返回 chunk 数"""
        self._chunks.clear()
        self._documents.clear()
        if not self.knowledge_dir.exists():
            logger.warning("知识库目录不存在：%s", self.knowledge_dir)
            return 0

        total_chunks = 0
        for file in sorted(self.knowledge_dir.glob("*")):
            if file.suffix.lower() not in {".md", ".txt"}:
                continue
            count = self.load_file(file)
            logger.info("加载文档 %s → %d 个分块", file.name, count)
            total_chunks += count
        return total_chunks

    def load_file(self, file: Path) -> int:
        """加载单个文档，返回新增 chunk 数"""
        text = file.read_text(encoding="utf-8")
        sections = _parse_md(text)
        count = 0
        for sec in sections:
            doc_type = self._infer_doc_type(file.name, sec["title"])
            metadata = {
                "source": file.name,
                "category": sec["title"],
                "doc_type": doc_type,
            }
            for chunk in self.chunker.split_text(sec["content"], metadata):
                self._chunks.append(chunk)
                count += 1
        if not sections and text.strip():
            # 无标题结构的纯文本
            metadata = {"source": file.name, "category": "综合", "doc_type": "general"}
            for chunk in self.chunker.split_text(text, metadata):
                self._chunks.append(chunk)
                count += 1
        self._documents.append(
            {
                "file_name": file.name,
                "chunks": count,
                "sections": len(sections),
                "doc_type": self._infer_doc_type(file.name, ""),
            }
        )
        return count

    # ---------- 对外查询 ----------
    @property
    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    @property
    def documents(self) -> List[Dict]:
        return list(self._documents)

    @property
    def corpus(self) -> List[str]:
        """全部 chunk 文本（供 Embedding fit 计算 IDF）"""
        return [c.content for c in self._chunks]


knowledge_service = KnowledgeService()
