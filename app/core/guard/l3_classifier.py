"""L3：轻量分类器（双阈值判定）

职责：消化 L2 的灰区。L2 判"像攻击但不确信"，L3 直接做四分类，
按概率两头卡阈值：足够正常就放行、足够攻击就拦、两头都不够才交给 L4。

双阈值为什么不对称（Module 3.1.4）：
- 正常类 ≥ 0.90 才放行（放行门槛更高）
- 攻击类 ≥ 0.85 即拦截
- 因为"漏报一个攻击"的代价高于"多问一次"，灰区宁可继续下沉

当前实现：字符 n-gram TF-IDF + 逻辑回归（CPU 亚毫秒，可解释、可秒级重训）。
选型结论（已替换原"微调 RoBERTa-wwm"设计）：本场景语义变体主要被 L2 模板库接走，
L3 只做灰区细分类；离线留出实验里微调增益无法验证，而权重/GPU/导出成本确定。
RoBERTa 四分类头保留为**可选升级位**（未实跑）：接口一致，换实现不动流水线。
"""
from typing import Dict, Optional

ATTACK_CATEGORIES = ("instruction_override", "data_leak", "role_switch")
NORMAL_LABEL = "normal"

PASS_NORMAL_PROB = 0.90   # 正常类概率 ≥ 0.90 → 放行
BLOCK_ATTACK_PROB = 0.85  # 攻击类概率 ≥ 0.85 → 拦截


class L3Classifier:
    def __init__(self, model=None):
        self._model = model

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(self, text: str) -> Dict:
        """输出四类概率 + 双阈值动作"""
        if self._model is None:
            return {"available": False, "action": "review", "reason": "L3 模型未训练/未加载"}

        try:
            proba = self._model.predict_proba([text])[0]
            classes = list(self._model.classes_)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "action": "review", "reason": f"L3 推理失败：{exc}"}

        probs = {c: float(p) for c, p in zip(classes, proba)}
        normal_p = probs.get(NORMAL_LABEL, 0.0)
        attack_p = max((p for c, p in probs.items() if c != NORMAL_LABEL), default=0.0)
        top_cat = max((c for c in probs if c != NORMAL_LABEL),
                      key=lambda c: probs[c], default=None)

        if normal_p >= PASS_NORMAL_PROB:
            action, reason = "pass", "L3 正常类高置信"
        elif attack_p >= BLOCK_ATTACK_PROB:
            action, reason = "block", "L3 攻击类高置信"
        else:
            action, reason = "review", "L3 灰区，转 L4 复核"

        return {
            "available": True,
            "action": action,
            "reason": reason,
            "probs": {k: round(v, 4) for k, v in probs.items()},
            "normal_prob": round(normal_p, 4),
            "attack_prob": round(attack_p, 4),
            "category": top_cat,
        }
