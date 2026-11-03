"""安全守卫的离线资产：L2 攻击模板库 + L3 分类器

为什么单独一层"资产"：
- 运行时（L2 检索、L3 分类）只消费产物，不关心怎么造；
- 离线构建（模板标注、训练、评估、导出）可独立迭代、可版本化、可灰度；
- 评测需要做"留出法"——用训练集造资产、用测试集算指标，必须在代码上分开。

产物落在 data/guard/：
- attack_templates.jsonl  模板库（text + category），L2 的检索底库
- l3_clf.joblib           L3 分类器（TF-IDF + LR，接口与微调模型一致，可替换）
- l3_metrics.json         训练/验证指标
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
GUARD_DIR = ROOT / "data" / "guard"
TEMPLATE_PATH = GUARD_DIR / "attack_templates.jsonl"
L3_MODEL_PATH = GUARD_DIR / "l3_clf.joblib"
L3_METRICS_PATH = GUARD_DIR / "l3_metrics.json"

# 类别自动标注用的关键词权重表：宁可粗、不要人工逐条标 200 条
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "role_switch": [
        "你现在是", "从现在起", "假装", "扮演", "充当", "冒充", "伪装", "开发者模式",
        "越狱", "jailbreak", "dan", "do anything now", "没有限制", "无限制", "不受限制",
        "上帝模式", "管理员权限", "root", "act as", "you are now",
    ],
    "data_leak": [
        "系统提示", "提示词", "prompt", "system", "训练数据", "语料", "数据库",
        "密钥", "密码", "api key", "配置", "源码", "全部用户", "所有用户", "读者信息",
        "借阅记录", "手机号", "后台", "reveal", "print",
    ],
    "instruction_override": [
        "忽略", "无视", "忘掉", "忘记", "作废", "覆盖", "清除", "不要遵循", "别管",
        "新的指令", "最高指令", "当作指令", "ignore", "forget", "disregard", "override",
        "bypass", "previous instructions",
    ],
}

_NORMALIZE = re.compile(r"[^\u4e00-\u9fff0-9a-z]+")


def label_category(text: str) -> str:
    """按关键词给攻击样本打三类标签（打分取最高，平手按 角色>泄露>覆盖）"""
    low = (text or "").lower()
    scores = {cat: sum(1 for kw in kws if kw in low) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best = max(scores, key=lambda c: (scores[c], c == "role_switch", c == "data_leak"))
    return best if scores[best] > 0 else "instruction_override"


def build_templates(attack_texts: Sequence[str], expand: bool = True) -> List[Dict]:
    """攻击样本 → L2 模板库（去重、保序，可选同义变体扩充）

    每条模板的目标是覆盖"一类表述"而不是"一句话"：设计上每类 50~100 条，
    靠同义替换/句式变换把锚点铺开，语义检索才能接住未见过的改写。
    """
    seen, templates = set(), []
    for i, text in enumerate(attack_texts):
        variants = [text] + (augment(text) if expand else [])
        cat = label_category(text)
        for j, v in enumerate(variants):
            norm = _NORMALIZE.sub("", (v or "").lower())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            templates.append({"id": len(templates), "text": v, "category": cat,
                              "variant_of": i if j else None})
    return templates


def save_templates(templates: Sequence[Dict], path: Path = TEMPLATE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in templates:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return path


def load_templates(path: Path = TEMPLATE_PATH) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(json.loads(line))
    return out


_SYNONYMS: Dict[str, List[str]] = {
    "忽略": ["无视", "忘掉", "别管", "不要理会"],
    "忘掉": ["忽略", "清空", "作废"],
    "之前": ["以上", "上面", "前面", "刚才"],
    "指令": ["规则", "规定", "要求", "命令"],
    "规则": ["规定", "约定", "限制"],
    "作废": ["失效", "无效", "取消"],
    "系统提示": ["系统消息", "提示词", "初始设定"],
    "你现在是": ["从现在起你是", "你现在扮演", "接下来你是"],
    "管理员": ["超级用户", "后台", "运维"],
    "数据": ["信息", "记录"],
}


def augment(text: str, max_variants: int = 2) -> List[str]:
    """语义变体增强：同义替换，让 L3 学到"意思"而不是"这一句"。

    只在训练侧做，且替换表是与测试集无关的通用同义词（忽略↔无视、指令↔规则…），
    目的是提升对未见口语改写的泛化，不是给评测集打补丁。
    """
    out: List[str] = []
    for src, cands in _SYNONYMS.items():
        if src in text:
            for cand in cands[:max_variants]:
                variant = text.replace(src, cand, 1)
                if variant != text:
                    out.append(variant)
    return list(dict.fromkeys(out))


def train_l3(samples: Sequence[Tuple[str, str]], random_state: int = 42):
    """训练 L3 分类器：字符 n-gram TF-IDF + 逻辑回归（4 类：normal + 3 类攻击）

    为什么先用它而不是直接上 RoBERTa：
    - 训练秒级、CPU 推理亚毫秒，能把整条链路先跑通并产出可复现的指标；
    - 对外接口统一为 predict_proba(4 类)，后续换任意分类模型只替换实现、不动流水线；
    - 口径必须诚实：这是"轻量分类器实跑版"，RoBERTa 微调版是可选升级位、未实跑。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [t for t, _ in samples]
    labels = [y for _, y in samples]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1,
            sublinear_tf=True, max_features=60000,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                   random_state=random_state)),
    ])
    model.fit(texts, labels)
    return model


def save_l3(model, metrics: Dict, path: Path = L3_MODEL_PATH) -> Path:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    L3_METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_l3(path: Path = L3_MODEL_PATH):
    """加载 L3 模型；不存在或依赖缺失时返回 None（流水线自动跳过该层）"""
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:  # noqa: BLE001
        return None
