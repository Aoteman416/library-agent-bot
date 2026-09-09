"""对话状态追踪（Dialog State Tracker）+ 槽位填充

解决的问题（对应审查清单 #2、#4）：
- 业务型对话（续借/预约）需要多轮槽位填充
- 槽位缺失时主动追问，而非直接查询
- 对话状态结构化存储，而非仅历史拼接

设计要点（面试可讲）：
- 每个会话维护一个 DialogueState 对象，记录意图/槽位/阶段
- 阶段：idle → collecting_slots → confirming → completed
- 支持混合填充：一个句子可能同时提供多个槽位
- 追问策略：优先追问缺失最多的槽位
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# 槽位定义
@dataclass
class Slot:
    name: str
    value: Optional[str] = None
    required: bool = True
    filled: bool = False


# 业务型对话的槽位模板
SEAT_RESERVATION_SLOTS = {
    "area": Slot("area", None, True, False),       # 区域
    "date": Slot("date", None, True, False),       # 日期
    "start_time": Slot("start_time", None, True, False),  # 开始时间
    "end_time": Slot("end_time", None, True, False),      # 结束时间
}

RENEW_SLOTS = {
    "book_id": Slot("book_id", None, True, False),
}


@dataclass
class DialogueState:
    """对话状态追踪器"""

    session_id: str
    intent: Optional[str] = None       # 当前识别到的业务意图
    phase: str = "idle"                # idle / collecting_slots / confirming / completed
    slots: Dict[str, Slot] = field(default_factory=dict)
    last_user_id: Optional[str] = None
    pending_confirm: Optional[Dict] = None  # 待确认操作：{action, payload, confirm_id, expires_at}
    pending_query: Optional[str] = None    # 用户发起的原始查询（登录升级后恢复用）
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # ---------- 槽位操作 ----------
    def start_collection(self, intent: str, template: Dict[str, Slot]) -> None:
        """开始收集槽位"""
        self.intent = intent
        self.phase = "collecting_slots"
        # 深拷贝槽位模板
        self.slots = {k: Slot(v.name, v.value, v.required, v.filled) for k, v in template.items()}
        self.updated_at = datetime.now()

    def fill_slot(self, name: str, value: str) -> bool:
        """填充单个槽位，返回是否完成所有必要槽位"""
        if name in self.slots:
            self.slots[name].value = value
            self.slots[name].filled = True
            self.updated_at = datetime.now()
        return self.is_complete()

    def is_complete(self) -> bool:
        """所有必要槽位是否已填充"""
        return all(s.filled for s in self.slots.values() if s.required)

    def missing_slots(self) -> List[str]:
        """返回缺失槽位的名称列表"""
        return [name for name, s in self.slots.items() if s.required and not s.filled]

    def next_question(self) -> str:
        """生成下一个追问"""
        missing = self.missing_slots()
        if not missing:
            return ""
        slot_labels = {
            "area": "请告诉我想预约的区域（如「3楼自习室」）",
            "date": "请告诉我预约日期（如「明天」或「2026-08-23」）",
            "start_time": "请告诉我开始时间（如「下午2点」或「14:00」）",
            "end_time": "请告诉我结束时间（如「下午4点」或「16:00」）",
            "book_id": "请告诉我要续借的图书编号，或说书名",
        }
        # 优先追问第一个缺失槽位
        return slot_labels.get(missing[0], f"请告诉我{missing[0]}")

    # ---------- 确认操作 ----------
    def set_pending_confirm(self, action: str, payload: Dict, ttl_seconds: int = 300) -> str:
        """设置待确认操作，返回 confirm_id"""
        import uuid
        confirm_id = uuid.uuid4().hex[:8]
        self.pending_confirm = {
            "action": action,
            "payload": payload,
            "confirm_id": confirm_id,
            "expires_at": datetime.now() + timedelta(seconds=ttl_seconds),
            "confirmed": False,
        }
        self.phase = "confirming"
        self.updated_at = datetime.now()
        return confirm_id

    def is_confirm_expired(self) -> bool:
        if not self.pending_confirm:
            return False
        return datetime.now() > self.pending_confirm["expires_at"]

    def clear_pending_confirm(self) -> None:
        self.pending_confirm = None
        if self.phase == "confirming":
            self.phase = "idle"
        self.updated_at = datetime.now()

    def confirm(self) -> Optional[Dict]:
        """确认并返回待执行的操作 payload"""
        if not self.pending_confirm or self.is_confirm_expired():
            return None
        self.pending_confirm["confirmed"] = True
        self.phase = "completed"
        result = self.pending_confirm
        self.pending_confirm = None
        self.updated_at = datetime.now()
        return result

    def cancel(self) -> None:
        """取消当前流程"""
        self.phase = "idle"
        self.slots = {}
        self.pending_confirm = None
        self.pending_query = None
        self.intent = None
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "phase": self.phase,
            "slots": {
                k: {"name": s.name, "value": s.value, "filled": s.filled}
                for k, s in self.slots.items()
            },
            "pending_confirm": self.pending_confirm,
            "updated_at": self.updated_at.isoformat(),
        }
