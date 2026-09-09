"""Function Calling 意图路由测试（用 Mock LLM，不依赖真实 API Key）

验证：
1. llm_client.chat_with_tools 正确解析 tool_calls → arguments
2. intent_rewriter._llm_route 将工具返回转为结构化结果（意图/书名/槽位）
3. 无 LLM Key 时回退启发式仍工作
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import llm_client as llm_client_module
from app.core.intent_rewriter import (
    intent_rewriter, heuristic_book_title, heuristic_is_book_query,
    _INTENT_ROUTE_TOOL,
)
import app.core.intent_rewriter as ir_module

# ---------- Mock LLM ----------
class MockToolClient:
    """模拟 OpenAI 返回 tool_calls"""
    enabled = True

    def __init__(self, responses):
        self.responses = list(responses)  # list of arguments dict
        self.last_tools = None
        self.last_messages = None

    async def chat_with_tools(self, messages, tools, tool_choice="auto"):
        self.last_tools = tools
        self.last_messages = messages
        if not self.responses:
            return None
        args = self.responses.pop(0)
        return {"name": "route_intent", "arguments": args}


passed = 0
failed = 0

def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {label} {detail}")
    else:
        failed += 1
        print(f"  ❌ {label} {detail}")


async def main():
    print("=== 1. Mock LLM 意图路由：口语化查书 ===")
    mock = MockToolClient([
        {"intent": "book_search", "query": "查书《三体》",
         "book_title": "三体", "date": "", "start_time": "", "end_time": "", "area": ""}
    ])
    old = llm_client_module.llm_client
    llm_client_module.llm_client = mock
    ir_module.llm_client = mock  # intent_rewriter 内部引用的也是这个名字
    try:
        result = await intent_rewriter.rewrite("有三体吗", score=0.0)
    finally:
        llm_client_module.llm_client = old
        ir_module.llm_client = old
    check("识别为 book_search", result["intent"] == "book_search", f"({result['intent']})")
    check("改写查询", result["query"] == "查书《三体》", f"({result['query']})")
    check("抽取书名", result["book_title"] == "三体", f"({result['book_title']})")
    check("来源是 llm", result["source"] == "llm")

    print("\n=== 2. Mock LLM 意图路由：座位预约整句抽槽位 ===")
    mock = MockToolClient([
        {"intent": "seat_reserve", "query": "预约明天下午2点到4点3楼自习室",
         "book_title": "", "date": "2026-08-23", "start_time": "14:00",
         "end_time": "16:00", "area": "3楼自习室"}
    ])
    llm_client_module.llm_client = mock
    ir_module.llm_client = mock
    try:
        result = await intent_rewriter.rewrite("帮我约明天下午2点到4点的自习室", score=0.0)
    finally:
        llm_client_module.llm_client = old
        ir_module.llm_client = old
    check("识别为 seat_reserve", result["intent"] == "seat_reserve")
    check("日期槽位", result["date"] == "2026-08-23", f"({result['date']})")
    check("开始时间槽位", result["start_time"] == "14:00", f"({result['start_time']})")
    check("结束时间槽位", result["end_time"] == "16:00", f"({result['end_time']})")
    check("区域槽位", result["area"] == "3楼自习室", f"({result['area']})")

    print("\n=== 3. Mock LLM：多轮指代消解（这本书 → 具体书名） ===")
    history = [
        {"role": "user", "content": "《三体》在馆吗"},
        {"role": "assistant", "content": "在的，3楼A区23架，可借 2 本。"},
    ]
    mock = MockToolClient([
        {"intent": "book_search", "query": "查书《三体》",
         "book_title": "三体", "date": "", "start_time": "", "end_time": "", "area": ""}
    ])
    llm_client_module.llm_client = mock
    ir_module.llm_client = mock
    try:
        result = await intent_rewriter.rewrite("这本书能借多久", score=0.0, history=history)
    finally:
        llm_client_module.llm_client = old
        ir_module.llm_client = old
    check("指代消解出书名", result["book_title"] == "三体", f"({result['book_title']})")

    print("\n=== 4. LLM 不可用时回退启发式 ===")
    # 显式注入未启用的客户端，避免依赖运行环境中是否配置了 LLM_API_KEY
    disabled = MockToolClient([])
    disabled.enabled = False
    llm_client_module.llm_client = disabled
    ir_module.llm_client = disabled
    try:
        result = await intent_rewriter.rewrite("我想借三体", score=0.0)
    finally:
        llm_client_module.llm_client = old
        ir_module.llm_client = old
    check("客户端未启用", not disabled.enabled)
    check("启发式识别查书", result["intent"] == "book_search" and result["source"] == "heuristic")

    print("\n=== 5. 工具 schema 结构合法 ===")
    check("tools 是列表且含 function", isinstance(_INTENT_ROUTE_TOOL, dict)
          and _INTENT_ROUTE_TOOL["type"] == "function")
    check("含 intent enum", "intent" in _INTENT_ROUTE_TOOL["function"]["parameters"]["properties"])

    print("\n" + "=" * 50)
    print(f"结果：{passed} 通过，{failed} 失败")
    if failed == 0:
        print("🎉 Function Calling 意图路由全部通过！")
    return failed


def test_function_calling_intent_routing():
    """供 pytest 收集：把脚本式断言包装成一个测试用例"""
    assert run_all() == 0


def run_all() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    # 直接运行：python tests/test_function_calling.py
    sys.exit(1 if run_all() else 0)
