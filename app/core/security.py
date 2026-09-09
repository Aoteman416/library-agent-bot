"""Prompt 注入防护 + 安全过滤

解决的问题（对应审查清单 #12）：
- 防止用户通过 prompt 注入绕过系统约束
- 防止间接注入（知识库文档中的恶意内容）
- 敏感数据脱敏：对话中的身份证号、手机号、密码等

设计要点：
- 多层防御：关键词过滤 + 正则模式匹配 + 长度限制
- 输出层过滤：LLM 生成内容中的注入指令清除
- 审计日志：记录可疑输入
"""
import re
from typing import Dict, Optional

from .logger import logger

# Prompt 注入模式（不区分大小写）
_INJECTION_PATTERNS = [
    # 指令覆盖型
    r"忽略(以上|之前|前面)(所有|全部)?指令",
    r"不要(遵循|遵守)(以上|之前|前面)",
    r"forget (all )?(previous |above )?(instructions?|prompts?)",
    r"ignore (previous |above )?(instructions?|prompts?)",
    r"disregard (previous |above )?(instructions?|prompts?)",
    r"新(的|的)(指令|prompt|提示)",
    r"system\s*prompt",
    r"输出.*(系统提示|prompt|system)",
    # 数据泄露型
    r"你的(系统提示|系统prompt|初始prompt|训练数据)",
    r"what(?:'?s| is) your (system )?prompt",
    r"reveal (your |the )?(system )?prompt",
    r"print.*system.*prompt",
    # 角色切换型
    r"你现在是",
    r"you are now",
    r"jailbreak",
    r"dan mode",
    r"do anything now",
]

# 编译正则
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# 敏感数据脱敏模式（不使用 \b，因为中文环境下无单词边界）
_SENSITIVE_PATTERNS = [
    (re.compile(r"\d{17}[\dXx]"), "[身份证号]"),  # 身份证
    (re.compile(r"1[3-9]\d{9}"), "[手机号]"),  # 手机号
    (re.compile(r"密码[：:]\s*\S+"), "密码：[已隐藏]"),
    (re.compile(r"password[=:]\s*\S+", re.IGNORECASE), "password=[已隐藏]"),
]


class SecurityGuard:
    """安全守卫：注入检测 + 敏感数据脱敏"""

    def __init__(self):
        self._blocked_count = 0

    def check_injection(self, text: str) -> Dict:
        """
        检测用户输入是否包含注入指令
        
        返回: {
            "blocked": bool,
            "matched_pattern": str | None,
            "severity": "low" | "medium" | "high",
        }
        """
        for pattern in _COMPILED_PATTERNS:
            match = pattern.search(text)
            if match:
                self._blocked_count += 1
                matched = match.group(0)
                logger.warning("Prompt injection blocked: pattern=%s, text=%s", matched[:50], text[:100])
                return {
                    "blocked": True,
                    "matched_pattern": matched,
                    "severity": "high",
                    "message": "您的问题包含可疑内容，已被安全系统拦截。如有正常需求请换个方式提问。",
                }

        # 长度限制（单轮不超 2000 字符）
        if len(text) > 2000:
            return {
                "blocked": True,
                "matched_pattern": "too_long",
                "severity": "medium",
                "message": "输入过长（超过 2000 字符），请精简您的问题。",
            }

        # 重复字符检测（如"啊啊啊啊啊啊啊"可能是攻击）
        if re.search(r"(.)\1{30,}", text):
            return {
                "blocked": True,
                "matched_pattern": "repeated_chars",
                "severity": "medium",
                "message": "输入包含大量重复字符，请检查后重试。",
            }

        return {"blocked": False, "matched_pattern": None, "severity": "low"}

    def sanitize_input(self, text: str) -> str:
        """输入清洗：去除控制字符、限制长度"""
        # 移除控制字符（除换行和制表符）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # 限制长度
        text = text[:2000]
        return text

    def mask_sensitive_data(self, text: str) -> str:
        """敏感数据脱敏（用于日志和存储）"""
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def clean_llm_output(self, text: str) -> str:
        """LLM 输出清洗：移除注入尝试"""
        for pattern in _COMPILED_PATTERNS:
            text = pattern.sub("[安全拦截]", text)
        return text

    @property
    def blocked_count(self) -> int:
        return self._blocked_count


# 全局单例
security_guard = SecurityGuard()
