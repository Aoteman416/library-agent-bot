"""多轮查询改写器：指代消解 + 上下文注入

解决的问题（对应审查清单 #3）：
- 用户第二轮问「那这本书能借多久？」→ 将「这本书」还原为「《三体》」
- 用户问「帮我预约」→ 结合上下文补全缺失信息
- 查询改写后再进入 RAG 检索管道

设计要点（面试可讲）：
- 利用对话历史构建实体映射：book_title=三体, area=3楼自习室
- 指代消解：检测「这本书」「那里」「刚才的」等指代词
- 输出改写后的 query + 注入的上下文实体
"""
import re
from typing import Dict, List, Optional, Tuple


# 指代词模式
_REFERENCE_PATTERNS = {
    "book": [
        r"(这本书|那本书|这本|那本|刚才那本|之前那本|这本《([^》]+)》)",
        r"(它|该书|这本书|那本书)",
    ],
    "place": [
        r"(那里|那儿|那里的|刚才的地方|之前的位置)",
    ],
    "time": [
        r"(那时候|那个时间|刚才|之前|上次)",
    ],
}

# 槽位关键词 → 槽位名
_SLOT_HINTS = {
    "区域": "area", "地方": "area", "位置": "area", "哪": "area",
    "日期": "date", "时间": "time", "几点": "time",
    "开始": "start", "结束": "end", "到": "end",
}


class QueryRewriter:
    """多轮查询改写器"""

    def __init__(self):
        self._entity_memory: Dict[str, Dict] = {}  # session_id → entities

    def extract_entities_from_history(
        self, session_id: str, history: List[Dict[str, str]]
    ) -> Dict:
        """
        从对话历史中提取实体信息
        
        返回: {
            "books": ["三体", "C++ Primer"],
            "areas": ["3楼自习室"],
            "dates": ["2026-08-23"],
            "times": ["14:00"],
        }
        """
        entities = {"books": [], "areas": [], "dates": [], "times": []}

        for msg in history:
            content = msg.get("content", "")
            # 书名：《XXX》模式
            for match in re.finditer(r"《([^》]+)》", content):
                title = match.group(1)
                if title not in entities["books"]:
                    entities["books"].append(title)

            # ISO 日期
            for match in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", content):
                entities["dates"].append(match.group(1))

            # 时间
            for match in re.finditer(r"\b(\d{1,2}:\d{2})\b", content):
                entities["times"].append(match.group(1))

        self._entity_memory[session_id] = entities
        return entities

    def rewrite(
        self, session_id: str, query: str, history: List[Dict[str, str]]
    ) -> Tuple[str, Dict]:
        """
        改写用户查询，解决指代问题
        
        返回: (rewritten_query, injected_context)
        """
        entities = self.extract_entities_from_history(session_id, history)
        rewritten = query
        context_injected = {}

        # 1. 书名指代消解
        if entities["books"]:
            # 查找指代词
            for pattern in _REFERENCE_PATTERNS["book"]:
                match = re.search(pattern, query)
                if match:
                    # 用最近的书名替换
                    latest_book = entities["books"][-1]
                    # 替换指代词
                    ref = match.group(0)
                    rewritten = rewritten.replace(ref, f"《{latest_book}》")
                    context_injected["resolved_book"] = latest_book
                    break

        # 2. 检查是否需要上下文注入
        if self._is_context_dependent(query):
            # 注入历史上下文
            if entities["books"]:
                context_injected["book_context"] = entities["books"][-1]
            if entities["areas"]:
                context_injected["area_context"] = entities["areas"][-1]
            if entities["dates"]:
                context_injected["date_context"] = entities["dates"][-1]

        return rewritten, context_injected

    @staticmethod
    def _is_context_dependent(query: str) -> bool:
        """判断查询是否依赖上下文"""
        context_markers = [
            "这本", "那本", "这本书", "那本书", "它",
            "那里", "那儿", "这个地方", "刚才",
            "那时候", "那个时间", "之前", "上次",
            "那", "那边", "刚才的",
        ]
        return any(marker in query for marker in context_markers)

    def clear_entities(self, session_id: str) -> None:
        if session_id in self._entity_memory:
            del self._entity_memory[session_id]


query_rewriter = QueryRewriter()
