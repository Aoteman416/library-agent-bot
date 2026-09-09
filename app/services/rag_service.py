"""RAG 服务：检索 + 生成编排

设计要点（面试可讲）：
- 完整 RAG 流程：文档分块 → Embedding 建索引 → 向量检索 → 重排 → 生成
- 生成双模式：
  1. LLM 模式：配置 LLM_API_KEY 后走真实大模型
  2. 离线模板模式：无 Key 时用检索到的知识块直接组装回答，
     保证系统开箱即用，且回答始终有知识库支撑（可讲清楚幻觉控制思路）
- stream_answer 以 SSE 事件流返回，兼容流式体验
"""
from typing import AsyncIterator, Dict, List

from ..config import settings
from ..core.llm_client import llm_client
from ..core.logger import logger
from ..rag.embeddings import create_embedder
from ..rag.prompt_builder import build_messages
from ..rag.retriever import Retriever
from ..rag.vector_store import create_store
from ..utils.helpers import chunk_text
from .knowledge_service import knowledge_service


class RagService:
    """RAG 检索增强生成服务"""

    def __init__(self):
        self.retriever: Retriever | None = None
        self._indexed = False

    # ---------- 索引构建 ----------
    def build_index(self, force_rebuild: bool = False) -> None:
        """从知识库文档构建检索索引

        force_rebuild=False 时，若向量库中已有数量一致的分块（如 Chroma 落盘
        的索引），直接复用，避免每次启动重复跑一遍 Embedding。
        """
        logger.info(
            "开始构建 RAG 索引（embedding=%s, store=%s）",
            settings.EMBEDDING_MODE,
            settings.VECTOR_STORE_MODE,
        )
        n_chunks = knowledge_service.load_directory()
        chunks = knowledge_service.chunks

        embedder = create_embedder(settings.EMBEDDING_MODE)
        store = create_store(settings.VECTOR_STORE_MODE)
        # 稀疏实现用它算 IDF；稠密实现为空操作，仅保持调用一致
        embedder.fit(knowledge_service.corpus)

        if not force_rebuild and store.size == len(chunks) and store.size > 0:
            logger.info("复用已持久化索引：%d 个分块（需重建请调用 reindex）", store.size)
        else:
            store.clear()
            if chunks:
                # 批量编码：稠密模型走 batch 推理，比逐条 encode 快一个数量级
                vectors = embedder.embed_batch([c.content for c in chunks])
                if hasattr(store, "add_batch"):
                    store.add_batch(chunks, vectors)
                else:
                    for chunk, vector in zip(chunks, vectors):
                        store.add(chunk, vector)
            logger.info("RAG 索引构建完成：%d 个文档分块", store.size)

        # 重排序器（Cross-Encoder 精排，对应 v2 3.1 Step3）
        from ..rag.reranker import create_reranker
        reranker = create_reranker(settings.RERANKER_ENABLED)
        logger.info("重排序器：%s", reranker.name)

        self.retriever = Retriever(embedder, store, reranker)
        self._indexed = True
        logger.info(
            "RAG 索引就绪：%d 个文档分块，向量库容量=%d",
            n_chunks,
            store.size,
        )

    @property
    def indexed(self) -> bool:
        return self._indexed

    @property
    def kb_size(self) -> int:
        if not self.retriever:
            return 0
        return self.retriever.store.size

    # ---------- 检索 ----------
    def retrieve(self, query: str) -> List[Dict]:
        """向量初检 + 重排"""
        if not self.retriever:
            self.build_index()
        return self.retriever.retrieve(
            query,
            top_k=settings.TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
        )

    # ---------- 生成 ----------
    @staticmethod
    def _template_answer(question: str, docs: List[Dict]) -> str:
        """离线模板模式：由检索到的知识块组装回答"""
        if not docs:
            return (
                "抱歉，我暂时没有找到关于这个问题的准确信息。\n"
                "建议您咨询图书馆总服务台，或输入「人工」转接人工客服。"
            )
        lines = ["根据图书馆知识库，我为您找到以下相关信息：\n"]
        for i, item in enumerate(docs, 1):
            chunk = item["chunk"]
            lines.append(f"{i}. {chunk.content.strip()}")
            lines.append(f"   （来源：{chunk.metadata.get('source', '未知')}）")
        lines.append("\n以上信息来自图书馆知识库，如有疑问可咨询总服务台。")
        return "\n".join(lines)

    def _sources(self, docs: List[Dict]) -> List[Dict[str, str]]:
        return [
            {
                "title": item["chunk"].metadata.get("source", "未知"),
                "category": item["chunk"].metadata.get("category", "综合"),
                "score": str(item["score"]),
            }
            for item in docs
        ]

    # 对外暴露：供 chat_service 直接调用
    def get_sources(self, docs: List[Dict]) -> List[Dict[str, str]]:
        """获取检索结果的引用来源（对外接口）"""
        return self._sources(docs)

    def template_answer(self, question: str, docs: List[Dict]) -> str:
        """离线模板模式答案生成（对外接口）"""
        return self._template_answer(question, docs)

    async def generate_async(
        self, question: str, history: List[Dict[str, str]]
    ) -> Dict:
        """非流式生成：返回 {reply_type, content, sources, mode}"""
        docs = self.retrieve(question)
        answer: str | None = None
        mode = "template"
        if llm_client.enabled:
            messages = build_messages(question, docs, history)
            answer = await llm_client.chat(messages)
            if answer:
                mode = "llm"
        if not answer:
            answer = self._template_answer(question, docs)
            mode = "template"
        return {
            "reply_type": "text",
            "content": answer,
            "sources": self._sources(docs),
            "mode": mode,
        }

    async def stream_answer(
        self, question: str, history: List[Dict[str, str]]
    ) -> AsyncIterator[Dict]:
        """流式生成：逐块产出事件字典
        事件格式：{"event": str, "data": dict}
        """
        docs = self.retrieve(question)
        sources = self._sources(docs)

        yield {"event": "retrieved", "data": {"count": len(docs)}}

        answer: str | None = None
        if llm_client.enabled:
            messages = build_messages(question, docs, history)
            try:
                parts: List[str] = []
                async for delta in llm_client.chat_stream(messages):
                    parts.append(delta)
                    yield {"event": "token", "data": {"content": delta}}
                if parts:
                    answer = "".join(parts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 流式失败，降级到离线模板：%s", exc)
                answer = None

        if not answer:
            answer = self._template_answer(question, docs)
            # 模板模式也按小片断流式输出，保持前端流式体验一致
            for piece in chunk_text(answer, size=12):
                yield {"event": "token", "data": {"content": piece}}

        yield {"event": "sources", "data": {"documents": sources}}
        yield {
            "event": "done",
            "data": {
                "reply_type": "text",
                "answer": answer,
                "sources": sources,
                "mode": "llm" if llm_client.enabled and answer else "template",
            },
        }


rag_service = RagService()
