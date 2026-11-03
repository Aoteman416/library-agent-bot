"""意图路由评测 - 分层 K 折交叉验证（实验1 增强版）

评测对象：app.services.chat_service.IntentClassifier（加权关键词分类器，一号信号）

重要说明（面试可讲，务必如实）：
- IntentClassifier 是**确定性规则分类器**（硬编码 INTENT_WEIGHTS），无训练过程，
  因此本脚本的 K 折**不是**「训练-验证」切分，而是用**固定模型**在分层子样本上
  估计评测指标的数据层面稳定性 / 置信区间（等价于 repeated stratified hold-out）。
- 采用按意图分层抽样（StratifiedKFold 思路），保证每折各类比例大致一致。
- 报告：每折 accuracy / macro-F1 + 均值 ± 标准差，以及全量（合并各折）混淆矩阵。
- 若各折指标完全相同（std=0），同样如实呈现：说明在分层抽样下该确定性分类器
  表现稳定，波动主要来自极少量边界样本在不同折间的归属。

指标：accuracy / 每类 P-R-F1 / macro-F1 / unknown 率 / 混淆矩阵
数据集：data/eval/intent_eval.jsonl（{id,text,intent}），intent ∈ 8 路

用法：
    python eval/run_intent_cv.py                 # 默认 5 折
    python eval/run_intent_cv.py --folds 10      # 10 折
    python eval/run_intent_cv.py --seed 42       # 固定随机种子
"""
import argparse
import json
import random
import statistics
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


def per_class_metrics(y_true, y_pred, classes):
    cm = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per = {}
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[t][cls] for t in classes if t != cls)
        fn = sum(cm[cls][p] for p in classes if p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[cls] = (prec, rec, f1)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    macro_f1 = sum(v[2] for v in per.values()) / len(classes)
    return acc, macro_f1, per, cm


def stratified_folds(cases, folds, seed):
    """按意图分层抽样生成 K 折（每折作为一次测试集）"""
    rnd = random.Random(seed)
    by_class = defaultdict(list)
    for i, c in enumerate(cases):
        by_class[c["intent"]].append(i)
    folds_idx = [[] for _ in range(folds)]
    for cls, idxs in by_class.items():
        rnd.shuffle(idxs)
        for pos, idx in enumerate(idxs):
            folds_idx[pos % folds].append(idx)
    all_idx = set(range(len(cases)))
    splits = []
    for k in range(folds):
        test = folds_idx[k]
        train = list(all_idx - set(test))
        splits.append((train, test))
    return splits


def format_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line, "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |", line]
    for row in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    out.append(line)
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="意图分类 分层K折交叉验证")
    ap.add_argument("--eval-set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--folds", type=int, default=5, help="折数 K")
    ap.add_argument("--seed", type=int, default=0, help="分层抽样随机种子")
    args = ap.parse_args(argv)

    cases = load(args.eval_set)
    clf = IntentClassifier()
    splits = stratified_folds(cases, args.folds, args.seed)

    fold_acc, fold_f1 = [], []
    per_fold = []
    full_cm = {t: {p: 0 for p in INTENTS} for t in INTENTS}

    print(f"\n数据集: {args.eval_set.name}  样本数: {len(cases)}  折数: {args.folds}  seed: {args.seed}")
    print("=" * 64)
    for k, (train, test) in enumerate(splits, 1):
        y_true, y_pred = [], []
        for i in test:
            pred, _, _ = clf.classify(cases[i]["text"])
            y_true.append(cases[i]["intent"])
            y_pred.append(pred)
        acc, macro_f1, per, cm = per_class_metrics(y_true, y_pred, INTENTS)
        fold_acc.append(acc)
        fold_f1.append(macro_f1)
        per_fold.append({"fold": k, "n_test": len(test),
                         "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4)})
        for t in INTENTS:
            for p in INTENTS:
                full_cm[t][p] += cm[t][p]
        print(f"  Fold {k}: n={len(test):>3}  accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")

    mean_acc, std_acc = statistics.mean(fold_acc), statistics.pstdev(fold_acc)
    mean_f1, std_f1 = statistics.mean(fold_f1), statistics.pstdev(fold_f1)
    print("-" * 64)
    print(f"  均值 ± 标准差:  accuracy={mean_acc:.4f} ± {std_acc:.4f}"
          f"   macro-F1={mean_f1:.4f} ± {std_f1:.4f}")
    print("=" * 64)

    print("\n各折汇总：")
    print(format_table(
        ["Fold", "n", "accuracy", "macro-F1"],
        [[pf["fold"], pf["n_test"], pf["accuracy"], pf["macro_f1"]] for pf in per_fold],
    ))

    print("\n全量混淆矩阵 (行=真实 / 列=预测)：")
    header = "真实\\预测".ljust(14) + "".join(c[:9].rjust(10) for c in INTENTS)
    print(header)
    for t in INTENTS:
        row = t[:9].ljust(14) + "".join(str(full_cm[t][p]).rjust(10) for p in INTENTS)
        print(row)

    out = {
        "num_cases": len(cases),
        "folds": args.folds,
        "seed": args.seed,
        "per_fold": per_fold,
        "accuracy_mean": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4),
        "macro_f1_mean": round(mean_f1, 4),
        "macro_f1_std": round(std_f1, 4),
        "confusion_matrix": full_cm,
    }
    (ROOT / "data" / "eval" / "intent_cv_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n交叉验证报告已写入 data/eval/intent_cv_report.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
