"""意图路由评测（实验1）

评测对象：app.services.chat_service.IntentClassifier（加权关键词分类器，一号信号）
说明：生产链路是「关键词(一号信号) + LLM概率(二号信号)」融合，本评测离线只跑确定的
      关键词信号（LLM 信号需线上环境），口径在回填故事时如实标注。

指标：accuracy / 每类 precision-recall-F1 / macro-F1 / unknown 率 / 混淆矩阵
数据集：data/eval/intent_eval.jsonl（{id,text,intent}），intent ∈ 8 路

用法：
    python eval/run_intent_eval.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.chat_service import IntentClassifier  # noqa: E402

DEFAULT_SET = ROOT / "data" / "eval" / "intent_eval.jsonl"
INTENTS = [
    "seat_reserve", "borrow_manage", "book_search", "human_transfer",
    "policy_faq", "resource_guide", "navigation", "unknown",
]


def load(path: Path):
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, default=DEFAULT_SET)
    args = ap.parse_args(argv)

    cases = load(args.eval_set)
    clf = IntentClassifier()

    y_true, y_pred = [], []
    for c in cases:
        pred_intent, _score, _ch = clf.classify(c["text"])
        y_true.append(c["intent"])
        y_pred.append(pred_intent)

    n = len(y_true)
    classes = INTENTS

    # 混淆矩阵 & 每类计数
    cm = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # 每类 P/R/F1
    per_class = {}
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[t][cls] for t in classes if t != cls)
        fn = sum(cm[cls][p] for p in classes if p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = (prec, rec, f1, tp, fp, fn)

    macro_f1 = sum(v[2] for v in per_class.values()) / len(classes)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

    true_unknown = sum(1 for t in y_true if t == "unknown")
    pred_unknown = sum(1 for p in y_pred if p == "unknown")
    true_unknown_rate = true_unknown / n
    pred_unknown_rate = pred_unknown / n

    # 输出
    print(f"\n样本数: {n}")
    print(f"整体准确率 accuracy      : {acc:.4f}")
    print(f"宏平均 F1 macro-F1      : {macro_f1:.4f}")
    print(f"真实 unknown 占比        : {true_unknown_rate:.4f}")
    print(f"预测 unknown 占比        : {pred_unknown_rate:.4f}")
    print()
    print(f"{'意图':<16}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    print("-" * 50)
    for cls in classes:
        p, r, f1, tp, fp, fn = per_class[cls]
        print(f"{cls:<16}{p:>7.3f}{r:>7.3f}{f1:>7.3f}{tp:>5}{fp:>5}{fn:>5}")

    # 混淆矩阵
    print("\n混淆矩阵 (行=真实 / 列=预测)：")
    header = "真实\\预测".ljust(14) + "".join(c[:9].rjust(10) for c in classes)
    print(header)
    for t in classes:
        row = t[:9].ljust(14) + "".join(str(cm[t][p]).rjust(10) for p in classes)
        print(row)

    # 错误样本
    print("\n错误预测样本：")
    for c, t, p in zip(cases, y_true, y_pred):
        if t != p:
            print(f"  #{c['id']:>3} 真实={t:<14} 预测={p:<14} | {c['text']}")

    # 落盘
    out = {
        "num_cases": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "true_unknown_rate": round(true_unknown_rate, 4),
        "pred_unknown_rate": round(pred_unknown_rate, 4),
        "per_class": {k: {"precision": round(v[0], 4), "recall": round(v[1], 4),
                          "f1": round(v[2], 4), "tp": v[3], "fp": v[4], "fn": v[5]}
                     for k, v in per_class.items()},
        "confusion_matrix": cm,
    }
    (ROOT / "data" / "eval" / "intent_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 data/eval/intent_report.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
