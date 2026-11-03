"""四层安全守卫流水线：L1 正则 → L2 语义向量 → L3 分类器 → L4 大模型复核

核心不是"四层堆起来"，而是三层短路规则（Module 3.1.6）：
1. 任一层给出「拦截」→ 立即返回，不再往下走（省掉后面 99% 的成本）；
2. 任一层给出「高置信放行」→ 立即放行（L2 就能放掉大部分正常请求）；
3. 只有「不确定」才下沉，且越往下越贵、处理的流量越少。

预期流量分布：L1 拦掉字面攻击 → L2 再处理大部分变体 → L3 只吃灰区 → L4 只吃极小灰区。
这条"成本梯度"就是面试里能讲的核心：安全不是把请求都丢给大模型，而是让 90% 的请求在第一层被解决。
"""
import os
import time
from typing import Dict, List, Optional, Sequence

from ..logger import logger
from . import assets
from .l1_regex import scan as l1_scan
from .l2_semantic import L2SemanticGuard
from .l3_classifier import L3Classifier
from .l4_llm import L4LLMJudge
from .normalize import views

BLOCK_MESSAGE = "您的问题包含可疑内容，已被安全系统拦截。如有正常需求请换个方式提问。"


class GuardPipeline:
    def __init__(
        self,
        templates: Optional[Sequence[Dict]] = None,
        classifier=None,
        l2: Optional[L2SemanticGuard] = None,
        l4: Optional[L4LLMJudge] = None,
        enable_l2: Optional[bool] = None,
        enable_l3: Optional[bool] = None,
        enable_l4: Optional[bool] = None,
    ):
        self._templates = templates
        self._l2 = l2
        self._l3 = L3Classifier(classifier)
        self._l4 = l4
        # 环境开关：默认全开，但某一层资产缺失时会自动跳过（不能因为没训练就整条链路挂掉）
        self.enable_l2 = _env_flag("GUARD_ENABLE_L2", True) if enable_l2 is None else enable_l2
        self.enable_l3 = _env_flag("GUARD_ENABLE_L3", True) if enable_l3 is None else enable_l3
        self.enable_l4 = _env_flag("GUARD_ENABLE_L4", True) if enable_l4 is None else enable_l4
        self._stats = {"total": 0, "blocked": 0,
                       "L1": 0, "L2": 0, "L3": 0, "L4": 0, "pass": 0}

    # ---------- 懒加载资产 ----------
    @property
    def l2(self) -> L2SemanticGuard:
        if self._l2 is None:
            templates = self._templates
            if templates is None:
                templates = assets.load_templates()
            self._l2 = L2SemanticGuard(templates)
        return self._l2

    @property
    def l4(self) -> L4LLMJudge:
        if self._l4 is None:
            self._l4 = L4LLMJudge()
        return self._l4

    # ---------- 主流程 ----------
    def evaluate(self, text: str) -> Dict:
        t0 = time.perf_counter()
        text = text or ""
        layers: List[Dict] = []

        # ---------- L0 前置：长度 / 重复字符（沿用旧规则，不算四层之一） ----------
        if len(text) > 2000:
            return self._result(True, "L0", "too_long", 1.0, t0, layers,
                                "输入过长（超过 2000 字符），请精简您的问题。", "medium")
        import re as _re
        if _re.search(r"(.)\1{30,}", text):
            return self._result(True, "L0", "repeated_chars", 1.0, t0, layers,
                                "输入包含大量重复字符，请检查后重试。", "medium")

        v = views(text)

        # ---------- L1 ----------
        t = time.perf_counter()
        hit = l1_scan(v)
        layers.append({"layer": "L1", "action": "block" if hit else "pass",
                       "category": (hit or {}).get("category"),
                       "latency_ms": _ms(t)})
        if hit:
            logger.warning("L1 拦截 类别=%s 命中=%s 视图=%s", hit["category"], hit["matched"], hit["view"])
            return self._result(True, "L1", hit["category"], 1.0, t0, layers,
                                BLOCK_MESSAGE, "high", matched_pattern=hit["matched"],
                                detail=hit)

        # ---------- L2 ----------
        if self.enable_l2 and self.l2.available:
            t = time.perf_counter()
            d2 = self.l2.decide(text)
            layers.append({"layer": "L2", "action": d2["action"],
                           "score": d2.get("score"), "sim": d2.get("sim"),
                           "category": d2.get("category"), "latency_ms": _ms(t)})
            if d2["action"] == "block":
                return self._result(True, "L2", d2.get("category"), d2.get("sim", 0.0),
                                    t0, layers, BLOCK_MESSAGE, "high",
                                    matched_pattern=d2.get("template"), detail=d2)
            if d2["action"] == "pass":
                return self._result(False, "L2", None, 1.0 - d2.get("sim", 0.0),
                                    t0, layers, "", "low", detail=d2)
        else:
            layers.append({"layer": "L2", "action": "skip", "reason": "模板库为空或未启用"})

        # ---------- L3 ----------
        if self.enable_l3 and self._l3.available:
            t = time.perf_counter()
            d3 = self._l3.predict(text)
            layers.append({"layer": "L3", "action": d3["action"],
                           "normal_prob": d3.get("normal_prob"),
                           "attack_prob": d3.get("attack_prob"),
                           "category": d3.get("category"), "latency_ms": _ms(t)})
            if d3["action"] == "block":
                return self._result(True, "L3", d3.get("category"), d3.get("attack_prob", 0.0),
                                    t0, layers, BLOCK_MESSAGE, "high", detail=d3)
            if d3["action"] == "pass":
                return self._result(False, "L3", None, d3.get("normal_prob", 0.0),
                                    t0, layers, "", "low", detail=d3)
        else:
            layers.append({"layer": "L3", "action": "skip", "reason": "模型未加载"})

        # ---------- L4 ----------
        if self.enable_l4:
            t = time.perf_counter()
            d4 = self.l4.judge(text)
            layers.append({"layer": "L4", "action": "block" if d4.get("is_attack") else "pass",
                           "ok": d4.get("ok"), "latency_ms": _ms(t)})
            if d4.get("is_attack"):
                return self._result(True, "L4", d4.get("category"), 0.5, t0, layers,
                                    BLOCK_MESSAGE, "high", detail=d4)
            if d4.get("ok"):
                return self._result(False, "L4", None, 0.5, t0, layers, "", "low", detail=d4)
            # 失败/超时：放行但记录 + 待人工审核（不能把正常用户挡在门外）
            res = self._result(False, "L4", None, 0.5, t0, layers, "", "medium", detail=d4)
            res["audit"] = {"pending_review": True, "reason": d4.get("reason", "L4 未给出结论")}
            return res

        layers.append({"layer": "L4", "action": "skip", "reason": "未配置 LLM 或未启用"})
        return self._result(False, "L3", None, 0.5, t0, layers, "", "medium", detail={"note": "灰区无 L4，记录放行"})

    # ---------- 结果封装 ----------
    def _result(self, blocked: bool, layer: str, category: Optional[str], confidence: float,
                t0: float, layers: List[Dict], message: str, severity: str,
                matched_pattern: Optional[str] = None, detail: Optional[Dict] = None) -> Dict:
        self._stats["total"] += 1
        if blocked:
            self._stats["blocked"] += 1
            self._stats[layer] = self._stats.get(layer, 0) + 1
        else:
            self._stats["pass"] += 1
        if blocked and category:
            key = f"cat:{category}"
            self._stats[key] = self._stats.get(key, 0) + 1
        return {
            # ---- 兼容旧的 check_injection 返回结构 ----
            "blocked": blocked,
            "matched_pattern": matched_pattern,
            "severity": severity,
            "message": message,
            # ---- 新增：统一裁决信息 ----
            "layer": layer,
            "category": category,
            "confidence": round(float(confidence), 4),
            "latency_ms": _ms(t0),
            "layers": layers,
            "detail": detail or {},
        }

    @property
    def stats(self) -> Dict:
        return dict(self._stats)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_PIPELINE: Optional[GuardPipeline] = None


def get_pipeline() -> GuardPipeline:
    """进程级单例：模型/模板只加载一次（L2 的 embedding 模型很贵）"""
    global _PIPELINE
    if _PIPELINE is None:
        templates = assets.load_templates()
        classifier = assets.load_l3()
        _PIPELINE = GuardPipeline(templates=templates, classifier=classifier)
        logger.info("安全守卫初始化：模板 %d 条，L2=%s L3=%s L4=%s",
                    len(templates), bool(templates), bool(classifier), _PIPELINE.l4.enabled)
    return _PIPELINE
