"""意图「模型概率」信号可行性探针（轻量分类器版）

背景：QLoRA（S-P1-005）为设计态未训练，但"模型概率能否救回被关键词误判进
unknown 的样本"这个问题可以先用字符 n-gram TF-IDF + 逻辑回归验证。

三种协议：
  A  2000 条内 5 折分层 CV —— 偏高不可报（1700 条为同种子改写族，折间共享字面片段）
  B  300 条人工 anchor 训练 → 1700 条合成改写评测 —— 跨分布、零重叠，**主口径**
  C  1700 条合成训练 → 300 条人工评测 —— 反向验证

用法：python eval/run_intent_probe.py
输出：data/eval/intent_probe_report.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "eval" / "intent_eval_2000.jsonl"
OUT = ROOT / "data" / "eval" / "intent_probe_report.json"


def build():
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
    return vec, clf


def score(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if p == "unknown" and t == "unknown")
    fp = sum(1 for t, p in zip(y_true, y_pred) if p == "unknown" and t != "unknown")
    fn = sum(1 for t, p in zip(y_true, y_pred) if p != "unknown" and t == "unknown")
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "pred_unknown_rate": round(sum(1 for p in y_pred if p == "unknown") / len(y_pred), 4),
        "unknown_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "unknown_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "unknown_tp": tp,
        "unknown_fp": fp,
        "unknown_fn": fn,
    }


def fit_predict(texts, y, tr, te):
    vec, clf = build()
    x_tr = vec.fit_transform([texts[i] for i in tr])
    x_te = vec.transform([texts[i] for i in te])
    clf.fit(x_tr, [y[i] for i in tr])
    return list(clf.predict(x_te))


def main():
    rows = [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()]
    texts = [r["text"] for r in rows]
    y = [r["intent"] for r in rows]
    src = [r["source"] for r in rows]

    report = {
        "model": "char_wb 1-3gram TFIDF + LogisticRegression(C=5, balanced)",
        "dataset": str(DATA.relative_to(ROOT)).replace("\\", "/"),
        "baseline_keyword": {
            "accuracy": 0.6220, "macro_f1": 0.6536,
            "pred_unknown_rate": 0.4025, "unknown_precision": 0.3056, "unknown_fp": 559,
        },
    }

    # A: 5 折（仅参考）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.array(["?"] * len(y), dtype=object)
    folds = []
    for tr, te in skf.split(texts, y):
        pred = fit_predict(texts, y, tr, te)
        folds.append(round(accuracy_score([y[i] for i in te], pred), 4))
        oof[te] = pred
    report["protocol_a_5fold_all2000"] = score(y, list(oof))
    report["protocol_a_5fold_all2000"]["per_fold_accuracy"] = folds
    report["protocol_a_5fold_all2000"]["note"] = "泄漏性乐观，不可作为对外成绩"

    # B: 人工训 → 合成评（主口径）
    tr = [i for i, s in enumerate(src) if s == "original"]
    te = [i for i, s in enumerate(src) if s != "original"]
    report["protocol_b_train_anchor_test_llm"] = score(
        [y[i] for i in te], fit_predict(texts, y, tr, te))
    report["protocol_b_train_anchor_test_llm"]["note"] = "跨分布零重叠，主口径"

    # C: 合成训 → 人工评
    report["protocol_c_train_llm_test_anchor"] = score(
        [y[i] for i in tr], fit_predict(texts, y, te, tr))

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for k in ("protocol_a_5fold_all2000", "protocol_b_train_anchor_test_llm", "protocol_c_train_llm_test_anchor"):
        r = report[k]
        print(f"{k}: acc={r['accuracy']} macroF1={r['macro_f1']} "
              f"pred_unk={r['pred_unknown_rate']} unkP={r['unknown_precision']} "
              f"unkR={r['unknown_recall']} tp={r['unknown_tp']} fp={r['unknown_fp']}")
    print("report ->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
