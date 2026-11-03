"""构建安全守卫离线资产：L2 攻击模板库 + L3 分类器

流程（与运行时严格分离）：
1. 读红队集 data/eval/security_eval.jsonl（140 攻击 + 60 正常）
2. 攻击 → 按关键词粗标三类 → 写入 data/guard/attack_templates.jsonl（L2 检索底库）
3. 攻击 + 正常（红队正常样本 + 意图集 300 条 + 检索集 200 条问题）→ 训练 L3 四分类器
4. 5 折交叉验证给出真实指标，模型与指标落盘

为什么正常样本要扩到 400+ 条：
L3 是"正常 vs 三类攻击"的四分类，正常类样本太少会让模型把所有非常见问法都判成攻击，
误报率会直接爆掉；用业务侧真实问法（借还/检索/座位/办证）做负样本，才是安全的负样本分布。

用法：
    python scripts/build_guard_assets.py
    python scripts/build_guard_assets.py --no-cv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.guard import assets  # noqa: E402

SECURITY_SET = ROOT / "data" / "eval" / "security_eval.jsonl"
INTENT_SET = ROOT / "data" / "eval" / "intent_eval.jsonl"
QA_SET = ROOT / "data" / "eval" / "qa_eval.jsonl"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(json.loads(line))
    return out


def build_benign() -> list:
    """正常样本：红队正常集 + 意图集问题 + 检索集问题（去重）"""
    texts = []
    for c in load_jsonl(SECURITY_SET):
        if c.get("label") == "benign":
            texts.append(c["text"])
    texts += [c["text"] for c in load_jsonl(INTENT_SET) if c.get("text")]
    texts += [c["query"] for c in load_jsonl(QA_SET) if c.get("query")]
    return list(dict.fromkeys(texts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="跳过交叉验证（更快）")
    args = ap.parse_args(argv)

    cases = load_jsonl(SECURITY_SET)
    attacks = [c["text"] for c in cases if c.get("label") == "attack"]
    benign = build_benign()

    # ---------- L2 模板库 ----------
    templates = assets.build_templates(attacks)
    assets.save_templates(templates)
    by_cat = {}
    for t in templates:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    print(f"L2 模板库：{len(templates)} 条  {by_cat}")

    # ---------- L3 分类器 ----------
    samples = [(t["text"], t["category"]) for t in templates]
    samples += [(v, t["category"]) for t in templates for v in assets.augment(t["text"])]
    samples += [(t, "normal") for t in benign]
    print(f"L3 训练集：{len(samples)} 条（攻击 {len(templates)} / 正常 {len(benign)}）")

    model = assets.train_l3(samples)
    metrics = {"train_size": len(samples), "classes": list(model.classes_)}

    if not args.no_cv:
        from sklearn.metrics import classification_report, f1_score
        from sklearn.model_selection import StratifiedKFold

        import numpy as np

        texts = [t for t, _ in samples]
        labels = [y for _, y in samples]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for tr, te in skf.split(texts, labels):
            m = assets.train_l3([(texts[i], labels[i]) for i in tr])
            pred = m.predict([texts[i] for i in te])
            accs.append((pred == np.array([labels[i] for i in te])).mean())
            f1s.append(f1_score([labels[i] for i in te], pred, average="macro"))
        metrics["cv_accuracy_mean"] = round(float(np.mean(accs)), 4)
        metrics["cv_accuracy_std"] = round(float(np.std(accs)), 4)
        metrics["cv_macro_f1_mean"] = round(float(np.mean(f1s)), 4)
        metrics["cv_macro_f1_std"] = round(float(np.std(f1s)), 4)
        print(f"5 折交叉验证：acc={metrics['cv_accuracy_mean']}±{metrics['cv_accuracy_std']} "
              f"macro-F1={metrics['cv_macro_f1_mean']}±{metrics['cv_macro_f1_std']}")
        print(classification_report(labels, model.predict(texts), digits=3))

    assets.save_l3(model, metrics)
    print(f"模型已写入 {assets.L3_MODEL_PATH}")
    print(f"指标已写入 {assets.L3_METRICS_PATH}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
