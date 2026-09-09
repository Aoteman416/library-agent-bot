"""对话服务 V2：完整的多轮对话编排

解决审查清单问题：
- #1 意图路由：加权分类 + 混合意图处理
- #2 槽位填充：DST + 追问流程
- #3 多轮改写：指代消解 + 上下文注入
- #4 二次确认：幂等键 + 超时 + 撤销
- #5 登录升级：匿名→登录→恢复执行
- #6 权限矩阵：功能级访问控制
- #12 Prompt 注入防护
- #13 并发冲突：幂等操作
- #14 会话隔离：公共设备超时
- #15 敏感数据脱敏
- #17 FAQ 语义缓存
"""
from typing import AsyncIterator, Dict, List, Optional, Tuple

from ..core.cache import faq_cache
from ..config import settings
from ..core.dst import DialogueState, SEAT_RESERVATION_SLOTS, RENEW_SLOTS
from ..core.llm_client import llm_client
from ..core.nlu import parse_time, parse_date, parse_area, extract_entities
from ..core.permissions import permission_manager, auth_upgrade_flow
from ..core.query_rewriter import query_rewriter
from ..core.intent_rewriter import intent_rewriter
from ..core.security import security_guard
from ..core.session_manager import Session
from ..utils.helpers import chunk_text
from .admin_service import admin_service
from .book_service import book_service
from .rag_service import rag_service
from .seat_service import seat_service
from .user_service import user_service


# ================= 意图权重表（不变）=================
INTENT_WEIGHTS: Dict[str, List[Tuple[str, int]]] = {
    "human_transfer": [("人工", 5), ("投诉", 5), ("转接", 4), ("客服", 3), ("投诉电话", 5)],
    "book_search": [
        ("查书", 5), ("找书", 5), ("找一本", 4), ("查询", 3),
        ("索书号", 5), ("isbn", 4), ("借阅状态", 4), ("有没有在架", 4),
        ("那本书", 3), ("这本书", 3), ("想看", 3), ("有没有", 2),
    ],
    "borrow_manage": [
        ("续借", 5), ("还书", 5), ("借阅", 3), ("欠费", 4), ("逾期", 4),
        ("借了", 3), ("借书记录", 4), ("过期", 3), ("罚款", 3), ("欠", 2),
    ],
    "seat_reserve": [
        ("预约座位", 5), ("自习室", 4), ("研习间", 4), ("座位预约", 5), ("订座", 4),
        ("预约", 3), ("座位", 3), ("自习", 3), ("空位", 3),
    ],
    "policy_faq": [
        ("开放时间", 5), ("开放", 2), ("办证", 5), ("赔偿", 5), ("荐购", 4),
        ("能带", 4), ("咖啡", 4), ("借书规则", 5), ("借阅规则", 5),
        ("寒假", 3), ("暑假", 3), ("闭馆", 3),
    ],
    "resource_guide": [
        ("知网", 5), ("论文查重", 5), ("vpn", 4), ("数据库", 4),
        ("自助打印", 5), ("电子资源", 4), ("查重", 4),
    ],
    "navigation": [
        ("服务台在哪", 5), ("总服务台", 4), ("还书箱", 4),
        ("布局", 3), ("位置", 3), ("在哪", 3), ("哪里", 3),
    ],
}

FAQ_INTENTS = {"policy_faq", "resource_guide", "navigation"}
BUSINESS_INTENTS = {"human_transfer", "book_search", "borrow_manage", "seat_reserve"}
RAG_LOW_CONFIDENCE_THRESHOLD = 0.15

# 需要登录的功能
AUTH_REQUIRED_FEATURES = {
    "borrow_manage", "seat_reserve",
}


class IntentClassifier:
    """加权意图分类器"""

    def classify(self, text: str) -> Tuple[str, float, str]:
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for intent, rules in INTENT_WEIGHTS.items():
            total = 0
            for keyword, weight in rules:
                if keyword.lower() in text_lower:
                    total += weight
            if total > 0:
                scores[intent] = total

        if not scores:
            return "unknown", 0.0, "rag"

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_intent in BUSINESS_INTENTS:
            channel = "business"
        elif best_intent in FAQ_INTENTS:
            channel = "faq"
        else:
            channel = "rag"

        # 冲突消解：业务 vs FAQ
        second_intent = self._second_best(scores, best_intent)
        if second_intent and scores[second_intent] >= best_score * 0.8:
            if best_intent in BUSINESS_INTENTS and second_intent in FAQ_INTENTS:
                return second_intent, scores[second_intent], "faq"

        return best_intent, best_score, channel

    @staticmethod
    def _second_best(scores: Dict[str, float], exclude: str) -> Optional[str]:
        filtered = {k: v for k, v in scores.items() if k != exclude}
        if not filtered:
            return None
        return max(filtered, key=filtered.get)


class ChatServiceV2:
    """对话编排服务 V2：三通道 + 多轮 + DST + 安全"""

    def __init__(self):
        self.classifier = IntentClassifier()

    # ================= 主入口 =================
    async def process(
        self,
        session: Session,
        message: str,
        user_id: str = "guest",
        user_role: str = "student",
    ) -> AsyncIterator[Dict]:
        """处理一条消息的完整流程"""
        # 1. 安全检查
        security_check = security_guard.check_injection(message)
        if security_check["blocked"]:
            yield self._event("token", {"content": security_check["message"]})
            yield self._event("done", {
                "reply_type": "text",
                "answer": security_check["message"],
                "security_blocked": True,
            })
            return

        # 2. 会话状态检查：是否在槽位收集阶段
        state = session.state
        if state.phase == "collecting_slots":
            async for ev in self._fill_slots(session, message, user_id):
                yield ev
            return

        # 3. 会话状态检查：是否待确认
        if state.phase == "confirming":
            async for ev in self._handle_confirmation(session, message, user_id):
                yield ev
            return

        # 4. 意图分类（关键词）
        intent, score, channel = self.classifier.classify(message)
        admin_service.record_question(message)
        admin_service.record_channel(channel)

        # 4.5 LLM 意图理解增强：低置信度时理解口语化语义并改写查询
        history = session.history()
        rewrite = await intent_rewriter.rewrite(
            message, score=score,
            session_id=session.session_id, history=history,
        )
        if rewrite["intent"]:
            intent = rewrite["intent"]
            if intent in BUSINESS_INTENTS:
                channel = "business"
            elif intent in FAQ_INTENTS:
                channel = "faq"
            else:
                channel = "rag"
        # 改写后的查询（LLM 语义改写 or 启发式规范化）
        rewritten_message = rewrite["query"]
        rewrite_context = {}
        if rewrite.get("book_title"):
            rewrite_context["book_title"] = rewrite["book_title"]
            rewrite_context["resolved_book"] = rewrite["book_title"]
        # Function Calling 抽出的业务槽位（座位预约预填充）
        for _slot in ("date", "start_time", "end_time", "area"):
            if rewrite.get(_slot):
                rewrite_context[_slot] = rewrite[_slot]

        # 5. 多轮指代消解改写（叠加）
        if channel == "rag":
            msg, context = query_rewriter.rewrite(
                session.session_id, message, history
            )
            rewritten_message = msg
            context.update(rewrite_context)
        else:
            context = rewrite_context or {}

        # 6. 路由决策
        if channel == "business":
            # 权限检查
            perm = permission_manager.can_access(user_id, user_role, intent)
            if not perm["allowed"] and perm["require_login"]:
                async for ev in self._require_login_flow(session, message, intent):
                    yield ev
                return

            async for ev in self._route_business(
                session, intent, user_id, rewritten_message, context
            ):
                yield ev
            return

        if channel == "faq":
            async for ev in self._route_faq(
                session, intent, rewritten_message, context
            ):
                yield ev
            return

        # RAG 通道
        # 特殊：unknown 意图 + LLM 可用 → 走 LLM 闲聊通道
        if (
            intent == "unknown"
            and llm_client.enabled
            and settings.CHAT_FALLBACK_TO_LLM
        ):
            async for ev in self._route_chat_llm(session, rewritten_message, history):
                yield ev
            return

        async for ev in self._route_rag(
            session, rewritten_message, history, context
        ):
            yield ev

    # ================= 业务通道 =================
    async def _route_business(
        self, session: Session, intent: str, user_id: str,
        message: str, context: Dict
    ) -> AsyncIterator[Dict]:
        yield self._event("route", {"channel": "business", "intent": intent})

        # 座位预约：需要槽位填充
        if intent == "seat_reserve":
            async for ev in self._seat_reserve_flow(
                session, user_id, message, context
            ):
                yield ev
            return

        # 续借：需要槽位填充
        if intent == "borrow_manage":
            async for ev in self._borrow_flow(session, user_id, message):
                yield ev
            return

        # 其他业务：直接处理
        reply = self._handle_simple_business(intent, user_id, message, context)
        if reply:
            async for ev in self._stream_reply(reply, channel="business"):
                yield ev
            session.add("user", message)
            session.add("assistant", reply.get("content", ""))

    def _handle_simple_business(
        self, intent: str, user_id: str, message: str, context: Dict | None = None
    ) -> Optional[Dict]:
        context = context or {}
        if intent == "book_search":
            return self._search_book(message, book_title=context.get("book_title"))
        if intent == "human_transfer":
            return {
                "reply_type": "transfer",
                "content": "正在为您转接人工客服（服务时间 9:00-17:00），请稍候...\n非服务时间会自动转为留言工单。",
                "extra": {"排队位置": 1, "预计等待": "2分钟", "夜间降级": True},
            }
        return None

    # ================= 座位预约流程（含槽位填充）=================
    async def _seat_reserve_flow(
        self, session: Session, user_id: str, message: str,
        context: Dict | None = None,
    ) -> AsyncIterator[Dict]:
        state = session.state
        entities = extract_entities(message)

        # 如果不在收集阶段，初始化
        if state.phase != "collecting_slots" or state.intent != "seat_reserve":
            state.start_collection("seat_reserve", SEAT_RESERVATION_SLOTS)

        # Function Calling 预抽取的槽位（LLM 路由时已从整句抽出）
        fc_slots = context or {}
        for slot_name in ("area", "date", "start_time", "end_time"):
            if fc_slots.get(slot_name):
                state.fill_slot(slot_name, fc_slots[slot_name])

        # 填充已识别的实体
        slot_map = {
            "area": entities.get("area"),
            "date": entities.get("date"),
            "start_time": entities.get("start_time"),
            "end_time": entities.get("end_time"),
        }
        for slot_name, value in slot_map.items():
            if value and slot_name in state.slots:
                state.fill_slot(slot_name, value)

        # 检查是否完成
        if state.is_complete():
            # 槽位完成，进入确认阶段
            payload = {
                "area": state.slots["area"].value,
                "date": state.slots["date"].value,
                "start_time": state.slots["start_time"].value,
                "end_time": state.slots["end_time"].value,
                "user_id": user_id,
            }
            confirm_id = state.set_pending_confirm("seat_reserve", payload)
            yield self._event("route", {"channel": "business", "phase": "confirming"})
            yield self._event("token", {"content": f"请确认预约信息：\n区域：{payload['area']}\n日期：{payload['date']}\n时间：{payload['start_time']} - {payload['end_time']}\n\n回复「确认」完成预约，回复「取消」取消操作。"})
            yield self._event("confirm", {
                "confirm_id": confirm_id,
                "action": "seat_reserve",
                "payload": payload,
            })
            yield self._event("done", {"reply_type": "confirm", "awaiting": True})
            return

        # 槽位未完成，追问
        question = state.next_question()
        yield self._event("route", {"channel": "business", "phase": "collecting_slots"})
        yield self._event("token", {"content": f"为了帮您预约座位，我还需要一些信息：\n{question}"})
        yield self._event("slot_required", {
            "missing_slots": state.missing_slots(),
            "current_slots": {k: s.value for k, s in state.slots.items() if s.filled},
        })
        yield self._event("done", {"reply_type": "slot_fill", "awaiting": True})

    # ================= 续借流程（含槽位填充）=================
    async def _borrow_flow(
        self, session: Session, user_id: str, message: str
    ) -> AsyncIterator[Dict]:
        # 检查是否包含续借意图
        if "续借" in message:
            state = session.state
            if state.phase != "collecting_slots" or state.intent != "borrow_manage":
                state.start_collection("borrow_manage", RENEW_SLOTS)

            # 尝试从历史中找书名
            history = session.history()
            book_title = None
            for msg in reversed(history):
                import re
                m = re.search(r"《([^》]+)》", msg.get("content", ""))
                if m:
                    book_title = m.group(1)
                    break

            if book_title:
                state.fill_slot("book_id", book_title)

            if state.is_complete():
                book_id = state.slots["book_id"].value
                confirm_id = state.set_pending_confirm(
                    "renew", {"user_id": user_id, "book_id": book_id}
                )
                yield self._event("token", {
                    "content": f"请确认续借《{book_id}》？\n回复「确认」完成续借，回复「取消」取消操作。"
                })
                yield self._event("confirm", {
                    "confirm_id": confirm_id,
                    "action": "renew",
                })
                yield self._event("done", {"reply_type": "confirm", "awaiting": True})
                return

            yield self._event("token", {
                "content": "请告诉我要续借的书名或图书编号（如《三体》或 B001）。"
            })
            yield self._event("done", {"reply_type": "slot_fill", "awaiting": True})
            return

        # 非续借操作，查询借阅信息
        info = user_service.get_borrow_info(user_id)
        lines = [f"用户：{info['user_id']}"]
        if info["borrowed"]:
            lines.append("当前借阅：")
            for b in info["borrowed"]:
                lines.append(
                    f"  - 《{b['title']}》到期：{b['due']}（已续借{b['renew_count']}次）"
                )
        else:
            lines.append("当前无借阅。")
        lines.append(f"欠费：¥{info['fines']:.1f}")
        yield self._event("token", {"content": "\n".join(lines)})
        yield self._event("done", {"reply_type": "text", "answer": "\n".join(lines)})

    # ================= 确认流程处理 =================
    async def _handle_confirmation(
        self, session: Session, message: str, user_id: str
    ) -> AsyncIterator[Dict]:
        state = session.state

        if state.is_confirm_expired():
            state.clear_pending_confirm()
            yield self._event("token", {
                "content": "确认操作已超时（5分钟内有效），请重新发起操作。"
            })
            yield self._event("done", {"reply_type": "text", "answer": "确认已超时"})
            return

        # 用户确认
        if message in ("确认", "是", "好", "ok", "yes", "对", "可以"):
            result = state.confirm()
            if result:
                async for ev in self._execute_confirm(session, result, user_id):
                    yield ev
                return

        # 用户取消
        elif message in ("取消", "不", "否", "cancel", "no", "算了"):
            state.clear_pending_confirm()
            yield self._event("token", {"content": "已取消操作。"})
            yield self._event("done", {"reply_type": "text", "answer": "已取消"})
            return

        # 模糊回复
        else:
            yield self._event("token", {
                "content": "请明确回复「确认」或「取消」。"
            })
            yield self._event("done", {"reply_type": "confirm", "awaiting": True})

    async def _execute_confirm(
        self, session: Session, confirm_result: Dict, user_id: str
    ) -> AsyncIterator[Dict]:
        action = confirm_result["action"]
        payload = confirm_result["payload"]

        if action == "seat_reserve":
            result = seat_service.reserve(
                payload["user_id"], payload["area"],
                payload["date"], payload["start_time"],
                payload["end_time"],
            )
            if result["ok"]:
                content = f"✅ 预约成功！\n区域：{payload['area']}\n日期：{payload['date']}\n时间：{payload['start_time']} - {payload['end_time']}\n预约编号：{result['reservation']['reservation_id']}"
            else:
                content = f"❌ 预约失败：{result['message']}"

        elif action == "renew":
            result = user_service.renew(payload["user_id"], payload["book_id"])
            if result["ok"]:
                content = f"✅ 续借成功！新到期日：{result.get('new_due', '')}"
            else:
                content = f"❌ 续借失败：{result['message']}"
        else:
            content = "操作已执行。"

        yield self._event("token", {"content": content})
        yield self._event("done", {
            "reply_type": "text",
            "answer": content,
            "action_result": result,
        })
        session.add("assistant", content)

    # ================= 槽位填充处理 =================
    async def _fill_slots(
        self, session: Session, message: str, user_id: str
    ) -> AsyncIterator[Dict]:
        state = session.state

        # 尝试从消息中提取实体
        entities = extract_entities(message)
        filled = False
        slot_map = {
            "area": entities.get("area"),
            "date": entities.get("date"),
            "start_time": entities.get("start_time"),
            "end_time": entities.get("end_time"),
            "book_id": message if state.intent == "borrow_manage" else None,
        }
        for slot_name, value in slot_map.items():
            if value and slot_name in state.slots:
                state.fill_slot(slot_name, value)
                filled = True

        if state.is_complete():
            # 转为确认阶段
            if state.intent == "seat_reserve":
                payload = {k: s.value for k, s in state.slots.items()}
                payload["user_id"] = user_id
            else:
                payload = {k: s.value for k, s in state.slots.items()}
                payload["user_id"] = user_id

            confirm_id = state.set_pending_confirm(state.intent, payload)
            yield self._event("token", {
                "content": f"请确认以下信息：\n{self._format_confirm(state.intent, payload)}\n\n回复「确认」完成操作，回复「取消」取消操作。"
            })
            yield self._event("confirm", {"confirm_id": confirm_id, "payload": payload})
            yield self._event("done", {"reply_type": "confirm", "awaiting": True})
            return

        # 继续追问
        question = state.next_question()
        if filled:
            question = f"已收到信息，还需要：{question}"
        yield self._event("token", {"content": question})
        yield self._event("done", {"reply_type": "slot_fill", "awaiting": True})

    @staticmethod
    def _format_confirm(intent: str, payload: Dict) -> str:
        if intent == "seat_reserve":
            return (
                f"区域：{payload.get('area')}\n日期：{payload.get('date')}\n"
                f"时间：{payload.get('start_time')} - {payload.get('end_time')}"
            )
        elif intent == "borrow_manage":
            return f"图书：{payload.get('book_id')}"
        return str(payload)

    # ================= 登录引导流程 =================
    async def _require_login_flow(
        self, session: Session, message: str, intent: str
    ) -> AsyncIterator[Dict]:
        upgrade = auth_upgrade_flow.require_login(
            session.session_id, intent, {"message": message}
        )
        content = (
            f"您请求的功能「{intent}」需要先登录。\n\n"
            f"演示账号：\n"
            f"  学号：2024001 / 2024002\n"
            f"  密码：123456\n\n"
            f"登录成功后会自动为您处理请求。"
        )
        yield self._event("token", {"content": content})
        yield self._event("login_required", {
            "message": content,
            "feature": intent,
            "demo_accounts": upgrade["demo_accounts"],
        })
        yield self._event("done", {
            "reply_type": "login_required", "awaiting": True, "channel": "business",
        })

    # ================= LLM 闲聊通道（unknown 意图 + LLM 可用）=================
    async def _route_chat_llm(
        self, session: Session, message: str, history: List[Dict]
    ) -> AsyncIterator[Dict]:
        """
        用大模型做基础对话（问候、闲聊、简单问答），
        作为「知识库未命中」的兜底通道，让客服更像「智能体」而非检索器。
        """
        yield self._event("route", {"channel": "chat_llm", "intent": "chat"})

        # 构造 LLM 消息序列：系统提示 + 最近对话历史 + 当前问题
        messages = self._build_chat_messages(history, message)

        answer: Optional[str] = None
        try:
            parts: List[str] = []
            async for delta in llm_client.chat_stream(messages):
                parts.append(delta)
                yield self._event("token", {"content": delta})
            if parts:
                answer = "".join(parts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("闲聊通道 LLM 流式失败，降级非流式：%s", exc)
            answer = await llm_client.chat(messages)
            if answer:
                for piece in chunk_text(answer, size=12):
                    yield self._event("token", {"content": piece})

        if not answer:
            answer = (
                "抱歉，我暂时没太理解您的意思。\n"
                "您可以问我：查书、预约座位、续借、开放时间、知网使用等图书馆相关问题～"
            )
            for piece in chunk_text(answer, size=12):
                yield self._event("token", {"content": piece})

        yield self._event("done", {
            "reply_type": "text",
            "answer": answer,
            "channel": "chat_llm",
        })
        session.add("user", message)
        session.add("assistant", answer)

    @staticmethod
    def _build_chat_messages(
        history: List[Dict], message: str, max_turns: int = 6
    ) -> List[Dict[str, str]]:
        """构建闲聊通道 LLM 消息序列"""
        sys = settings.CHAT_SYSTEM_PROMPT
        # 保留最近 max_turns 轮，控制 token 预算
        recent = history[-max_turns:]
        messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        for m in recent:
            role = m.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": message})
        return messages

    # ================= FAQ 通道 =================
    async def _route_faq(
        self, session: Session, intent: str, message: str, context: Dict
    ) -> AsyncIterator[Dict]:
        yield self._event("route", {"channel": "faq_fast", "intent": intent})

        # 语义缓存查找
        cached = faq_cache.lookup(message)
        if cached:
            answer, sources = cached
            yield self._event("token", {"content": answer})
            if sources:
                yield self._event("sources", {"documents": sources})
            yield self._event("done", {
                "reply_type": "text",
                "answer": answer,
                "sources": sources,
                "cache_hit": True,
            })
            session.add("assistant", answer)
            return

        # FAQ 快速命中
        faq_reply = self._faq_fast_match(message)
        if faq_reply:
            faq_cache.store(message, faq_reply["content"], faq_reply.get("sources"))
            async for ev in self._stream_reply(faq_reply, channel="faq_fast"):
                yield ev
            session.add("assistant", faq_reply["content"])
            return

        # 降级到 RAG
        history = session.history()
        async for ev in self._route_rag(session, message, history, context):
            yield ev

    def _faq_fast_match(self, message: str) -> Optional[Dict]:
        """知识块精确匹配"""
        from .knowledge_service import knowledge_service
        chunks = knowledge_service.chunks
        if not chunks:
            return None

        text_lower = message.lower()
        best_chunk = None
        best_score = 0

        for chunk in chunks:
            content_lower = chunk.content.lower()
            score = 0
            # 简单关键词提取
            for n in [2, 3, 4]:
                for i in range(len(text_lower) - n + 1):
                    gram = text_lower[i:i + n]
                    if gram in content_lower and len(gram) > 1:
                        score += len(gram)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk and best_score >= 4:
            source = best_chunk.metadata.get("source", "未知")
            return {
                "reply_type": "text",
                "content": f"{best_chunk.content}\n\n[来源：{source}]",
                "sources": [{"title": source, "category": best_chunk.metadata.get("category", "综合")}],
            }
        return None

    # ================= RAG 通道 =================
    async def _route_rag(
        self, session: Session, message: str, history: List[Dict],
        context: Dict
    ) -> AsyncIterator[Dict]:
        yield self._event("route", {"channel": "rag"})

        docs = rag_service.retrieve(message)
        sources = rag_service.get_sources(docs)

        # 低置信度检查
        low_confidence = False
        if not docs or docs[0]["score"] < RAG_LOW_CONFIDENCE_THRESHOLD:
            low_confidence = True
            confidence = docs[0]["score"] if docs else 0.0
            admin_service.record_unresolved(
                message, session_id=session.session_id, confidence=confidence
            )

        # 生成答案
        answer = await self._generate_answer(message, history, docs, low_confidence)

        # 存入缓存
        if not low_confidence:
            faq_cache.store(message, answer, sources)

        # 流式输出
        for piece in chunk_text(answer, size=12):
            yield self._event("token", {"content": piece})
        yield self._event("sources", {"documents": sources})
        yield self._event("done", {
                "reply_type": "text",
                "answer": answer,
                "sources": sources,
                "low_confidence": low_confidence,
                "channel": "rag",
            })
        session.add("user", message)
        session.add("assistant", answer)

    async def _generate_answer(
        self, message: str, history: List[Dict],
        docs: List[Dict], low_confidence: bool
    ) -> str:
        # 低置信度：如果没检索到文档 → 直接兜底（由上层的 chat_llm 通道优先处理 unknown 场景）
        if low_confidence and not docs:
            return (
                "抱歉，我暂时没有找到关于这个问题的准确信息。\n"
                "建议：\n"
                "  1. 换个关键词试试（如「开放时间」「借阅规则」）\n"
                "  2. 输入「人工」转接人工客服\n"
                "  3. 通过图书馆官网或电话咨询"
            )
        # 有检索结果时，优先用 LLM 生成（配置了 Key 时）
        result = await rag_service.generate_async(message, history)
        return result.get("content", "")

    # ================= 通用工具 =================
    @staticmethod
    def _event(event: str, data: Dict) -> Dict:
        return {"event": event, "data": data}

    @staticmethod
    async def _stream_reply(reply: Dict, channel: str = "") -> AsyncIterator[Dict]:
        content = reply["content"]
        for piece in chunk_text(content, size=12):
            yield {"event": "token", "data": {"content": piece}}
        if reply.get("extra"):
            yield {"event": "card", "data": {
                "reply_type": reply["reply_type"],
                "extra": reply["extra"],
            }}
        if reply.get("sources"):
            yield {"event": "sources", "data": {"documents": reply["sources"]}}
        yield {"event": "done", "data": {
            "reply_type": reply["reply_type"],
            "answer": content,
            "sources": reply.get("sources"),
            "channel": channel,
        }}

    def _search_book(self, message: str, book_title: str | None = None) -> Optional[Dict]:
        # 优先用改写层抽出的书名，否则用原文
        query = book_title or message
        result = book_service.search(query)
        if result["total"] == 0:
            return {
                "reply_type": "text",
                "content": f"未找到与「{query}」相关的馆藏图书，建议换个关键词试试。",
            }
        book = result["books"][0]
        status_map = {"available": "可借", "unavailable": "不可借", "partial": "部分可借"}
        return {
            "reply_type": "card",
            "content": f"《{book['title']}》",
            "extra": {
                "作者": book["author"],
                "ISBN": book["isbn"],
                "位置": book["location"],
                "馆藏": f"{book['available']}/{book['total']} 可借（{status_map.get(book['status'], book['status'])}）",
            },
        }


# 全局单例
chat_service = ChatServiceV2()
