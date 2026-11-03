"""统一后处理层（对应流程图 Module 4.3，替代原 K）

所有通道输出都进这里，做五件事：
① 输出安全审核（清洗 LLM/模板输出中的注入残留）
② 按通道差异化（FAQ 加时效、RAG 加来源、业务加事务结果与失败原因、通用加拒答/转人工判断）
③ 日志与 trace（意图 / 结果 / 安全审核）
④ 满意度预测与转人工判断
⑤ 输出最终响应结构
"""
from typing import Dict, List, Optional

from .logger import logger
from .security import security_guard


class PostProcessor:
    """统一后处理：所有通道的最终回复都经此增强与落日志"""

    def finalize(
        self,
        *,
        intent_result=None,
        channel: str = "",
        reply_type: str = "text",
        answer: str = "",
        sources: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None,
        action_result: Optional[Dict] = None,
        low_confidence: bool = False,
    ) -> Dict:
        # ① 输出安全审核：清洗生成内容中的注入残留
        if answer:
            answer = security_guard.clean_llm_output(answer)

        # ② 通道差异化
        if channel == "faq":
            answer = self._faq_format(answer)
        elif channel == "rag":
            answer = self._rag_format(answer, sources)
        elif channel == "business":
            answer = self._business_format(answer, action_result)

        # ④ 满意度预测 + 转人工判断
        satisfaction = self._predict_satisfaction(answer)
        suggest_human = self._suggest_human(answer, action_result)

        # ③/⑤ 组装并落日志
        data: Dict = {
            "reply_type": reply_type,
            "answer": answer,
            "channel": channel,
            "sources": sources,
            "extra": extra,
            "action_result": action_result,
            "low_confidence": low_confidence,
            "security_reviewed": True,
            "satisfaction": satisfaction,
            "suggest_human": suggest_human,
            "intent": intent_result.intent if intent_result else None,
            "confidence": intent_result.confidence if intent_result else None,
        }
        logger.info(
            "统一后处理: channel=%s intent=%s conf=%.2f sat=%s human=%s",
            channel,
            data["intent"],
            data["confidence"] or 0,
            satisfaction,
            suggest_human,
        )
        return data

    # ---------- 通道差异化 ----------
    @staticmethod
    def _faq_format(answer: str) -> str:
        # FAQ 快答：标注时效，避免用户把回答当永久规定
        if answer and "（信息可能更新）" not in answer:
            return answer + "\n（温馨提示：开放时间、规定等可能随馆方通知调整，请以官网最新公告为准）"
        return answer

    @staticmethod
    def _rag_format(answer: str, sources: Optional[List[Dict]]) -> str:
        if not sources:
            return answer
        # RAG 已带 sources 事件；此处仅做轻量收尾（不重复来源）
        return answer

    @staticmethod
    def _business_format(answer: str, action_result: Optional[Dict]) -> str:
        if action_result and action_result.get("result_type") in (
            "retryable_failure",
            "non_retryable_failure",
        ):
            # 业务失败：明确失败原因 + 替代方案（具体文案由通道给出，这里兜底）
            if "失败" not in answer:
                answer = f"❌ {answer}"
        return answer

    # ---------- 满意度 / 转人工 ----------
    @staticmethod
    def _predict_satisfaction(answer: str) -> str:
        if not answer:
            return "unknown"
        neg = ("抱歉", "无法", "失败", "没有找到", "暂不支持", "错误")
        pos = ("成功", "已完成", "可以", "为您", "已预约", "已续借")
        if any(k in answer for k in neg):
            return "low"
        if any(k in answer for k in pos):
            return "high"
        return "medium"

    @staticmethod
    def _suggest_human(answer: str, action_result: Optional[Dict]) -> bool:
        if action_result and action_result.get("result_type") == "non_retryable_failure":
            return True
        if any(k in answer for k in ("转人工", "人工客服", "投诉")):
            return True
        return False


# 全局单例
post_processor = PostProcessor()
