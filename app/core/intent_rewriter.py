"""意图理解 + 查询改写层

解决口语化自然语言（无关键词）无法路由到正确通道的问题，例如：
- 「有三体吗」「三体有在架吗」「我想借三体」→ 识别为 book_search 并抽取书名
- 「查一下三体」→ 改写成规范的图书检索查询

设计要点（面试可讲）：
- 双通道：
  1. 启发式：无 LLM Key 也能工作的图书口语化识别（抽取书名、识别查询意图）
  2. LLM 增强：配置 LLM_API_KEY 后，用大模型理解整个句子的意图并改写查询，
     不局限于图书，所有低置信度问题都会经 LLM 改写后再进检索管道
- 低置信度才介入，避免干扰已明确命中的高置信意图
- 改写后的 query 同时服务于业务通道（查书）与 RAG 通道（检索）
"""
import json
import re
from typing import Dict, Optional, Tuple

from ..config import settings
from ..core.llm_client import llm_client
from ..core.logger import logger

# 口语化图书查询的动词/疑问词
_BOOK_QUERY_VERBS = [
    "有没有在架", "在架吗", "在不在", "有没有", "有吗", "还有吗", "还有没有",
    "想借", "借一本", "怎么借", "能不能借", "可以借", "是否可以借",
    "查一下", "找一下", "看看", "有没有这", "有这", "馆藏", "有",
]

# 明确属于其他意图的场景，抽书名时排除（避免把「开放时间」等误当书名）
_NON_BOOK_HINTS = [
    "开放时间", "开馆", "闭馆", "座位", "自习室", "研习间", "预约", "续借",
    "还书箱", "还书", "逾期", "欠费", "罚款", "赔偿", "办证", "图书证",
    "知网", "查重", "数据库", "vpn", "自助打印", "打印机", "咖啡", "食物",
    "矿泉水", "规则", "制度", "赔偿标准", "寒假", "暑假", "服务台在哪",
    "人工", "投诉", "客服",
]

# 图书查询场景下的无意义词，用于抽取真正的书名
_BOOK_STOPWORDS = [
    "图书馆", "有没有在架", "有在架吗", "在架吗", "在不在", "有没有", "有没有这",
    "有吗", "吗", "呢", "的", "请问", "一下", "这本书", "那本书", "这本书吗",
    "那本书吗", "这本", "那本", "这本书", "那本书", "书吗", "是不", "是不是",
    "我想", "我要", "想", "帮我", "帮", "怎么", "如何", "借一本", "有这", "有本",
    "还有", "还", "？", "?", "《", "》",
]

# 书名前的口语化查询动词（去掉后剩书名本体）
_TITLE_LEAD_VERBS = ["想借", "借一本", "找一本", "看看", "查一下", "找一下",
                     "借", "查", "找", "看", "有", "想"]

# LLM 意图标签枚举（与分类器意图名对齐）
_INTENT_LABELS = {
    "book_search", "borrow_manage", "seat_reserve", "human_transfer",
    "policy_faq", "resource_guide", "navigation", "unknown",
}

# ================= Function Calling 工具定义 =================
# 用 LLM Function Calling 做「意图路由 + 槽位抽取 + 查询改写」
# 相比让 LLM 自由输出 JSON，工具调用由模型原生保障 schema 合法性，更稳定
_INTENT_ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "route_intent",
        "description": (
            "判断用户意图、改写查询并抽取关键实体。"
            "用于图书馆客服：查书、借阅管理、座位预约等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "book_search", "borrow_manage", "seat_reserve",
                        "human_transfer", "policy_faq", "resource_guide",
                        "navigation", "unknown",
                    ],
                    "description": "用户真实意图",
                },
                "query": {
                    "type": "string",
                    "description": "改写后的规范查询。查书改写为「查书《书名》」；"
                                    "座位预约保留原文；普通咨询规范化语气词。",
                },
                "book_title": {
                    "type": "string",
                    "description": "书名（查书时必填，不含书名号）",
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式 YYYY-MM-DD（座位预约时填，如 2026-08-23）",
                },
                "start_time": {
                    "type": "string",
                    "description": "开始时间，格式 HH:MM（座位预约时填）",
                },
                "end_time": {
                    "type": "string",
                    "description": "结束时间，格式 HH:MM（座位预约时填）",
                },
                "area": {
                    "type": "string",
                    "description": "区域（座位预约时填，如 3楼自习室）",
                },
            },
            "required": ["intent", "query"],
        },
    },
}

_FUNCTION_CALL_SYSTEM = (
    "你是图书馆客服系统的意图路由模块。请调用 route_intent 工具："
    "理解用户真实意图，改写查询，并抽取关键实体。\n"
    "规则：\n"
    "- 想查/看/找某本书 → intent=book_search，query改写为「查书《书名》」，book_title=书名\n"
    "- 续借/还书/借阅记录 → intent=borrow_manage\n"
    "- 座位/自习室/研习间/预约 → intent=seat_reserve，抽取 date/start_time/end_time/area\n"
    "- 开放时间/办证/赔偿等制度咨询 → intent=policy_faq\n"
    "- 知网/数据库/打印等资源 → intent=resource_guide；位置/导航 → intent=navigation\n"
    "- 转人工/投诉 → intent=human_transfer\n"
    "- 与图书馆无关 → intent=unknown，query 不变\n"
    "- 若用户话语中存在指代（这本书/明天下午），结合对话上下文还原为具体值"
)


def _strip_quotes(text: str) -> str:
    """去掉书名号《》"""
    return text.strip().replace("《", "").replace("》", "")


def heuristic_book_title(text: str) -> Optional[str]:
    """启发式抽取书名：
    1. 优先取《书名号》
    2. 否则去掉查询动词 + 停用词，取剩余核心词
    """
    # 书名号优先
    m = re.search(r"《([^》]+)》", text)
    if m:
        return m.group(1).strip()

    cleaned = text
    for w in sorted(_BOOK_STOPWORDS, key=len, reverse=True):
        cleaned = cleaned.replace(w, "")
    # 去掉书名前的查询动词（如「借」「查」「想借」）
    for verb in sorted(_TITLE_LEAD_VERBS, key=len, reverse=True):
        if cleaned.startswith(verb):
            cleaned = cleaned[len(verb):]
            break
    cleaned = re.sub(r"\s+", "", cleaned)
    # 剩余一个有效词就当作书名
    if 1 <= len(cleaned) <= 20:
        return cleaned.strip()
    return None


def heuristic_is_book_query(text: str) -> bool:
    """判断是否为口语化图书查询"""
    # 明确非图书意图的，直接排除
    for hint in _NON_BOOK_HINTS:
        if hint in text:
            return False
    lower = text.lower()
    if "《" in text:
        return True
    for v in _BOOK_QUERY_VERBS:
        if v in lower:
            title = heuristic_book_title(text)
            if title:
                return True
    return False


class IntentRewriter:
    """意图理解 + 查询改写"""

    async def rewrite(
        self,
        query: str,
        score: float = 0.0,
        session_id: str = "",
        history: Optional[list] = None,
    ) -> Dict:
        """
        返回: {
          "intent": Optional[str],   # 重写出的意图（None 表示维持原判）
          "query": str,              # 改写后的查询
          "book_title": Optional[str],
          "date": Optional[str], "start_time": Optional[str],
          "end_time": Optional[str], "area": Optional[str],
          "confidence": float,
          "source": "llm" | "heuristic" | "none",
        }
        """
        # 1) 若配置了 LLM：Function Calling 意图路由（低置信度或明确查书/预约都会走）
        if llm_client.enabled and score <= 2.0:
            llm_result = await self._llm_route(query, history)
            if llm_result:
                logger.info(
                    "LLM Function Calling 路由: intent=%s query=%s title=%s slots=%s",
                    llm_result.get("intent"), llm_result.get("query"),
                    llm_result.get("book_title"),
                    {k: llm_result.get(k) for k in
                     ("date", "start_time", "end_time", "area")},
                )
                llm_result["source"] = "llm"
                return llm_result

        # 2) 启发式兜底（无 Key / LLM 失败也能工作）
        if heuristic_is_book_query(query):
            title = heuristic_book_title(query)
            rewritten = f"查书《{title}》" if title else query
            return {
                "intent": "book_search",
                "query": rewritten,
                "book_title": title,
                "confidence": 0.85,
                "source": "heuristic",
            }

        return {"intent": None, "query": query, "book_title": None,
                "date": None, "start_time": None, "end_time": None,
                "area": None, "confidence": score, "source": "none"}

    async def _llm_route(
        self, query: str, history: Optional[list]
    ) -> Optional[Dict]:
        """调用 LLM Function Calling 识别意图 + 抽取槽位 + 改写查询"""
        sys = _FUNCTION_CALL_SYSTEM
        user = query
        if history:
            recent = history[-4:]
            hist_lines = [
                f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
                for m in recent
            ]
            if hist_lines:
                sys += "\n\n【对话上下文】\n" + "\n".join(hist_lines)

        try:
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
            tool_call = await llm_client.chat_with_tools(
                messages, [_INTENT_ROUTE_TOOL], tool_choice="auto"
            )
            if not tool_call or tool_call["name"] != "route_intent":
                return None
            args = tool_call["arguments"] or {}

            intent = str(args.get("intent", "")).strip().lower()
            if intent not in _INTENT_LABELS:
                intent = "unknown"
            book_title = (args.get("book_title") or "").strip() or None
            rewritten = (args.get("query") or query).strip() or query

            def _clean_slot(v):
                if not v:
                    return None
                v = str(v).strip()
                return v or None

            return {
                "intent": intent,
                "query": rewritten,
                "book_title": _strip_quotes(book_title) if book_title else None,
                "date": _clean_slot(args.get("date")),
                "start_time": _clean_slot(args.get("start_time")),
                "end_time": _clean_slot(args.get("end_time")),
                "area": _clean_slot(args.get("area")),
                "confidence": 0.9,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM Function Calling 路由失败，降级到启发式：%s", exc)
            return None


intent_rewriter = IntentRewriter()