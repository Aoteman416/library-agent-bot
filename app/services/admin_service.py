"""后台管理服务：对话统计 + 通道追踪 + 反馈收集 + 知识回流

新增功能：
- channel_counts: 各通道命中次数（business/faq_fast/rag）
- unresolved 记录：RAG 低置信度问题 → 知识回流入口
- feedback 收集：用户反馈评分
- overview 返回通道占比 + 满意度，便于运营分析
"""
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class AdminService:
    """对话数据统计 + 运营闭环"""

    def __init__(self):
        self._question_counts: Counter = Counter()
        self._channel_counts: Counter = Counter()
        self._unresolved: List[Dict] = []  # 改为 dict，记录更多上下文
        self._feedbacks: List[Dict] = []
        self._pending_knowledge: List[Dict] = []  # 待回流知识

    def record_question(self, question: str) -> None:
        self._question_counts[question] += 1

    def record_channel(self, channel: str) -> None:
        self._channel_counts[channel] += 1

    def record_unresolved(
        self, question: str, session_id: str = "", confidence: float = 0.0
    ) -> None:
        """记录未解决问题，供后续知识回流"""
        self._unresolved.append({
            "question": question,
            "session_id": session_id,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",  # pending / resolved / rejected
        })

    def submit_feedback(
        self,
        session_id: str,
        rating: int,
        comment: str = "",
        is_helpful: bool = True,
    ) -> Dict:
        """收集用户反馈"""
        fb = {
            "session_id": session_id,
            "rating": rating,
            "comment": comment,
            "is_helpful": is_helpful,
            "timestamp": datetime.now().isoformat(),
        }
        self._feedbacks.append(fb)

        # 低分反馈自动加入待知识回流
        if rating <= 3:
            self._pending_knowledge.append({
                "source": "low_rating_feedback",
                "rating": rating,
                "comment": comment,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "pending_review",
            })

        return {"status": "received", "feedback_id": len(self._feedbacks)}

    def get_hot_questions(self, top_n: int = 10) -> List[Tuple[str, int]]:
        return self._question_counts.most_common(top_n)

    def get_channel_distribution(self) -> List[Tuple[str, int]]:
        return self._channel_counts.most_common()

    def get_unresolved(self, status: Optional[str] = None) -> List[Dict]:
        if status:
            return [u for u in self._unresolved if u["status"] == status]
        return list(self._unresolved)

    def get_feedbacks(self, limit: int = 50) -> List[Dict]:
        return list(self._feedbacks[-limit:])

    def get_pending_knowledge(self) -> List[Dict]:
        return list(self._pending_knowledge)

    def resolve_unresolved(self, index: int, answer: str) -> bool:
        """馆员补充答案后，标记为已解决"""
        if 0 <= index < len(self._unresolved):
            self._unresolved[index]["status"] = "resolved"
            self._unresolved[index]["curated_answer"] = answer
            self._unresolved[index]["resolved_at"] = datetime.now().isoformat()
            # 同时加入待回流队列
            self._pending_knowledge.append({
                "source": "curated_unresolved",
                "question": self._unresolved[index]["question"],
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "status": "pending_review",
            })
            return True
        return False

    def get_overview(self) -> Dict:
        total = sum(self._channel_counts.values()) or 1
        channel_pct = {
            ch: round(cnt / total * 100, 1)
            for ch, cnt in self._channel_counts.items()
        }
        avg_rating = 0.0
        if self._feedbacks:
            avg_rating = round(
                sum(f["rating"] for f in self._feedbacks) / len(self._feedbacks), 1
            )
        return {
            "total_questions": sum(self._question_counts.values()),
            "unique_questions": len(self._question_counts),
            "unresolved_count": len([
                u for u in self._unresolved if u["status"] == "pending"
            ]),
            "resolved_count": len([
                u for u in self._unresolved if u["status"] == "resolved"
            ]),
            "feedback_count": len(self._feedbacks),
            "avg_rating": avg_rating,
            "pending_knowledge_count": len(self._pending_knowledge),
            "hot_questions": self.get_hot_questions(5),
            "channel_distribution": self.get_channel_distribution(),
            "channel_percentage": channel_pct,
        }


admin_service = AdminService()
