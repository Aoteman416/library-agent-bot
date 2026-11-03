"""意图识别与通道路由（对应流程图 Module 4）

设计原则：识别归识别、路由归路由、后处理归后处理。
- IntentResult   ：融合「关键词分数（一号信号）+ 模型概率（二号信号）」后的校准结果
- RouteDecisionMaker：意图 → 通道映射（放配置，不硬编码）、多意图拆分、置信度澄清
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import settings


# 意图 → 通道映射（配置化：业务 / FAQ / 人工 / 未知，不硬编码在编排里）
INTENT_CHANNEL_MAP: Dict[str, str] = {
    "seat_reserve": "business",
    "borrow_manage": "business",
    "book_search": "business",
    "human_transfer": "human",       # 人工转接（Module 9）
    "policy_faq": "faq",
    "resource_guide": "faq",
    "navigation": "faq",
    "unknown": "unknown",            # 澄清 → 通用兜底
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    channel: str
    multi_intent: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_options: List[str] = field(default_factory=list)
    source: str = "keyword"   # keyword | llm | merged


class RouteDecisionMaker:
    """六步顺序的轻量实现：多信号融合 / 多意图 / 置信度澄清 / 意图→通道映射"""

    def __init__(self, channel_map: Optional[Dict[str, str]] = None):
        self.channel_map = channel_map or INTENT_CHANNEL_MAP

    def decide(
        self,
        kw_intent: str,
        kw_score: float,
        kw_scores: Optional[Dict[str, float]] = None,
        llm_intent: str = "",
        llm_confidence: float = 0.0,
    ) -> IntentResult:
        # 1) 多信号融合：LLM 意图（二号信号）覆盖关键词（一号信号）
        if llm_intent:
            intent = llm_intent
            source = "llm" if (not kw_intent or kw_intent == "unknown") else "merged"
            # LLM 给出意图即视为有依据：给基线置信，避免无关键词命中时误判为低置信
            confidence = max(self._norm(kw_score), 0.7)
        else:
            intent = kw_intent
            source = "keyword"
            confidence = self._norm(kw_score)

        channel = self.channel_map.get(intent, "rag")

        # 2) 多意图检测：关键词 top-2 同时超阈 → 标记 multi_intent（供上层拆分）
        multi: List[str] = []
        if kw_scores:
            best = max(kw_scores.values())
            if best > 0:
                for it, sc in kw_scores.items():
                    if 0.8 * best <= sc < best:
                        multi.append(it)

        # 3) 置信度澄清：低置信且非 unknown → 澄清 + 候选选项；极低 → 直接兜底
        needs_clarification = False
        options: List[str] = []
        if intent != "unknown" and confidence < settings.CLARIFY_CONFIDENCE_THRESHOLD:
            needs_clarification = True
            options = self._candidates(kw_scores, intent)

        return IntentResult(
            intent=intent,
            confidence=round(confidence, 3),
            channel=channel,
            multi_intent=multi,
            needs_clarification=needs_clarification,
            clarification_options=options,
            source=source,
        )

    @staticmethod
    def _norm(kw_score: float) -> float:
        """关键词加权和 → 伪置信度（仅用于阈值判断，非真实概率）"""
        if kw_score <= 0:
            return 0.0
        # 经验映射：权重和 2 ≈ 0.4，5 ≈ 0.7，>=10 ≈ 0.95
        return min(1.0, kw_score / 10.0 + 0.3)

    @staticmethod
    def _candidates(kw_scores: Optional[Dict[str, float]], intent: str) -> List[str]:
        if not kw_scores:
            return [intent] if intent != "unknown" else []
        ranked = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
        return [it for it, sc in ranked if sc > 0][:3]
