"""LLM 客户端（OpenAI 兼容接口）

设计要点（面试可讲）：
- 通过 httpx 异步调用 OpenAI 格式的 /chat/completions
- 支持流式（SSE）与非流式两种模式
- 未配置 API Key / 调用失败时返回 None，
  由上层降级到「离线模板生成」，保证系统始终可用
"""
from typing import AsyncIterator, Dict, List, Optional

import httpx

from ..config import settings
from .logger import logger


class LLMClient:
    """OpenAI 兼容的 LLM 客户端"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT
        # 是否启用真实 LLM（未配置 Key 时走离线模板模式）
        self.enabled = bool(self.api_key)
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """非流式生成，失败返回 None"""
        if not self.enabled:
            return None
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 调用失败，降级到离线模板：%s", exc)
            return None

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_choice: str = "auto",
    ) -> Optional[Dict]:
        """Function Calling：让 LLM 输出结构化参数。

        返回解析后的工具调用参数 dict：
            {"name": str, "arguments": dict}
        失败 / 未配置 Key / 模型未调用工具时返回 None。
        """
        if not self.enabled:
            return None
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return None
            call = tool_calls[0]
            import json as _json

            arguments = _json.loads(call["function"]["arguments"] or "{}")
            return {
                "name": call["function"]["name"],
                "arguments": arguments,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM Function Calling 失败，降级到离线：%s", exc)
            return None

    async def chat_stream(
        self, messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """流式生成，逐块产出内容增量；失败时抛出异常由上层降级"""
        if not self.enabled:
            raise RuntimeError("LLM 未启用")
        client = self._get_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                import json

                data = json.loads(payload)
                delta = data["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


# 全局单例
llm_client = LLMClient()
