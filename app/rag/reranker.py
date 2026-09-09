"""重排序器：Cross-Encoder 精排（对应需求分析 v2 3.1 Step3、3.2.4）

设计要点：
- 默认 NoopReranker（融合打分即最终排序），保证零依赖可用
- RERANKER_ENABLED=true 时启用 Cross-Encoder（BAAI/bge-reranker-large），
  对「查询-文档」对做精细相关性打分，显著提升 Top-N 精度
- sentence-transformers 缺失时自动降级回 Noop，服务不中断（与全局降级策略一致）
"""
from typing import Dict, List

from .chunker import Chunk


class BaseReranker:
    name = "base"

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        return candidates


class NoopReranker(BaseReranker):
    """不重排：保留传入的 score 顺序（融合打分结果）"""

    name = "noop"


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排序（bge-reranker-large 等）"""

    name = "cross-encoder"

    def __init__(self, model_name: str = "", batch_size: int = 32, device: str = ""):
        from ..config import settings

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底
            raise ImportError(
                "未安装 sentence-transformers，无法使用 Cross-Encoder 重排序。\n"
                "安装：pip install -r requirements-rag.txt\n"
                "或设置 RERANKER_ENABLED=false 关闭重排序。"
            ) from exc

        self.model_name = model_name or settings.RERANKER_MODEL
        self.batch_size = batch_size or settings.RERANKER_BATCH_SIZE
        self._model = CrossEncoder(self.model_name, device=device or settings.EMBEDDING_DEVICE)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return candidates
        pairs = [(query, c["chunk"].content) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        for cand, s in zip(candidates, scores):
            cand["score"] = round(float(s), 4)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates


def create_reranker(enabled: bool = False, **kwargs) -> BaseReranker:
    """工厂：按配置创建重排序器，失败自动降级为无重排"""
    from ..core.logger import logger

    if enabled:
        try:
            return CrossEncoderReranker(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-Encoder 重排序初始化失败，降级为无重排：%s", exc)
    return NoopReranker()
