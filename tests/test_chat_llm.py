"""LLM 闲聊通道测试（用 Mock LLM 验证 unknown 意图会走 chat_llm 通道）"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import llm_client as llm_client_module
import app.core.llm_client as llm_module
from app.core.session_manager import session_manager
from app.services import chat_service as chat_svc_module
from app.services.chat_service import ChatServiceV2


class MockStreamLLM:
    enabled = True

    def __init__(self, reply):
        self.reply = reply

    async def chat_stream(self, messages):
        for ch in self.reply:
            yield ch

    async def chat(self, messages):
        return self.reply


async def main():
    svc = ChatServiceV2()

    # --- 场景 1：unknown 意图 + LLM 可用 → 走 chat_llm ---
    mock = MockStreamLLM("你好！我是图书馆智能客服，很高兴为您服务。您可以问我关于查书、预约、续借等问题～")
    old = llm_client_module.llm_client
    llm_client_module.llm_client = mock
    llm_module.llm_client = mock
    chat_svc_module.llm_client = mock

    try:
        session = session_manager.get_or_create("test-chat", "user1")
        events = []
        async for ev in svc.process(session, "你好", "user1"):
            events.append(ev)
        names = [e["event"] for e in events]
        done = next((e["data"] for e in events if e["event"] == "done"), {})
        print(f"[你好]")
        print(f"  事件序列: {names}")
        print(f"  channel={done.get('channel')} type={done.get('reply_type')}")
        full = "".join(e["data"].get("content", "") for e in events if e["event"] == "token")
        print(f"  回答: {full}")
        assert done.get("channel") == "chat_llm", f"期望 chat_llm 通道，实际 {done.get('channel')}"
        assert "你好" in full or "客服" in full, "期望 LLM 回答"
        print("  ✅ unknown 意图走 LLM 闲聊通道")

        # --- 场景 2：业务意图仍然走 business 通道 ---
        session2 = session_manager.get_or_create("test-book", "user2")
        events2 = []
        async for ev in svc.process(session2, "有三体吗", "user2"):
            events2.append(ev)
        done2 = next((e["data"] for e in events2 if e["event"] == "done"), {})
        print(f"\n[有三体吗]")
        print(f"  channel={done2.get('channel')} type={done2.get('reply_type')}")
        assert done2.get("channel") == "business", "业务意图仍应走 business 通道"
        print("  ✅ 业务意图不受影响")

        # --- 场景 3：FAQ 意图仍然走 faq ---
        session3 = session_manager.get_or_create("test-faq", "user3")
        events3 = []
        async for ev in svc.process(session3, "图书馆开放时间", "user3"):
            events3.append(ev)
        done3 = next((e["data"] for e in events3 if e["event"] == "done"), {})
        print(f"\n[图书馆开放时间]")
        print(f"  channel={done3.get('channel')} type={done3.get('reply_type')}")
        assert done3.get("channel") in ("faq_fast", "rag"), f"FAQ 应走 faq/rag，实际 {done3.get('channel')}"
        print("  ✅ FAQ 意图不受影响")

        # --- 场景 4：多轮对话保持上下文 ---
        session4 = session_manager.get_or_create("test-multi", "user4")
        # 先塞几条历史
        session4.add("user", "你好")
        session4.add("assistant", "你好！我是图书馆智能客服。")
        mock.reply = "没问题！关于这个问题我可以帮您。您具体想了解什么？"
        events4 = []
        async for ev in svc.process(session4, "今天天气怎么样", "user4"):
            events4.append(ev)
        full4 = "".join(e["data"].get("content", "") for e in events4 if e["event"] == "token")
        print(f"\n[多轮：今天天气怎么样]")
        print(f"  回答: {full4}")
        assert "没问题" in full4, "多轮对话应保留上下文"
        print("  ✅ 多轮上下文保持")

    finally:
        llm_client_module.llm_client = old
        llm_module.llm_client = old
        chat_svc_module.llm_client = old

    print("\n🎉 LLM 闲聊通道全部验证通过！")


asyncio.run(main())
