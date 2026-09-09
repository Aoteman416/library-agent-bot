"""Prompt 构建器

设计要点（面试可讲）：
- 系统 Prompt 约束角色、回答规则、幻觉控制（引用来源 / 不编造）
- 上下文构建采用「检索文档 + 对话历史 + 当前问题」三段式
- 对话历史只保留最近 N 轮，控制 Token 预算
"""
from typing import Dict, List

SYSTEM_PROMPT = """你是一个专业、友善的图书馆智能客服助手。请严格遵循以下规则：

## 角色定位
- 你是图书馆的服务助手，只能回答与图书馆相关的问题
- 回答要简洁、准确、友好，使用简体中文

## 回答规则
1. 优先使用参考文档中的信息，不要编造馆藏数据或规章制度
2. 如果参考文档中没有相关信息，请回复：
   "抱歉，我暂时无法回答这个问题，建议您咨询图书馆总服务台或输入「人工」转接人工客服。"
3. 回答末尾标注信息来源，格式为 [来源：文档名]
4. 涉及操作步骤时使用有序列表；涉及数字、时间要给出精确值

## 禁止行为
- 不回答与图书馆无关的问题
- 不编造不存在的规定或数据
- 不索要或透露个人隐私信息

## 对话历史（最近的对话）
{chat_history}

## 参考文档
{context}

## 用户问题
{question}
"""


def format_chat_history(history: List[Dict[str, str]], max_turns: int = 6) -> str:
    """格式化对话历史（仅保留最近 max_turns 轮）"""
    lines = []
    for msg in history[-max_turns:]:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}：{msg['content']}")
    return "\n".join(lines) or "（无）"


def build_context(docs: List[Dict]) -> str:
    """将检索结果拼接为上下文文本"""
    parts = []
    for i, item in enumerate(docs, 1):
        chunk: "Chunk" = item["chunk"]
        source = chunk.metadata.get("source", "未知来源")
        parts.append(f"[文档{i}] {chunk.content}\n来源：{source}")
    return "\n\n".join(parts)


def build_messages(
    question: str,
    docs: List[Dict],
    history: List[Dict[str, str]],
    max_history_turns: int = 6,
) -> List[Dict[str, str]]:
    """构建完整 LLM 消息序列"""
    system = SYSTEM_PROMPT.format(
        chat_history=format_chat_history(history, max_history_turns),
        context=build_context(docs) if docs else "（未检索到相关信息）",
        question=question,
    )
    return [{"role": "system", "content": system}]
