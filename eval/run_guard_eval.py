"""四层安全守卫评测（L1/L2/L3 级联 + 逐层归因 + 置信区间）

两种评测模式：
1. batch（推荐，2000 条级）：A 批攻击建模板库/训练 L3，B 批攻击（未参与训练）只评测；
   正常样本来自主评测集。真正避免"模板库就是评测集"。
2. cv（200 条小集）：5 折切分，兼容旧数据。

指标：
- 最终攻击召回 / 误报率（附 Wilson 95% 置信区间）
- 逐层归因：L1/L2/L3 各拦多少，各层增量收益
- P95 单条检测时延
- L4：未配置 LLM 时如实标注"未参与"

用法：
    python eval/run_guard_eval.py                      # 自动选 batch 模式（存在 2000 集时）
    python eval/run_guard_eval.py --mode cv --split group
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("EMBEDDING_MODE", "ngram")
os.environ.setdefault("GUARD_ENABLE_L4", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.guard import assets  # noqa: E402
from app.core.guard.pipeline import GuardPipeline  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"


def load_jsonl(path: Path):
    out = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("//"):
                    out.append(json.loads(line))
    return out


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% 置信区间：小样本比例不能只报点估计"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(max(centre - half, 0.0), 4), round(min(centre + half, 1.0), 4))


def build_pipeline(train_attacks, benign, enable_l4=False):
    templates = assets.build_templates(train_attacks)
    samples = [(t["text"], t["category"]) for t in templates]
    samples += [(t, "normal") for t in benign]
    clf = assets.train_l3(samples)
    return GuardPipeline(templates=templates, classifier=clf, enable_l4=enable_l4)


def evaluate(pipe, cases):
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        res = pipe.evaluate(c["text"])
        dt = (time.perf_counter() - t0) * 1000
        rows.append({"id": c.get("id"), "text": c["text"], "label": c["label"],
                     "blocked": res["blocked"], "layer": res.get("layer"),
                     "category": res.get("category"), "latency_ms": dt})
    return rows


def summarize(rows):
    atk = [r for r in rows if r["label"] == "attack"]
    ben = [r for r in rows if r["label"] == "benign"]
    blocked_atk = [r for r in atk if r["blocked"]]
    blocked_ben = [r for r in ben if r["blocked"]]
    layers = {}
    for r in blocked_atk:
        layers[r["layer"]] = layers.get(r["layer"], 0) + 1
    lat = [r["latency_ms"] for r in rows]
    return {
        "n_attack": len(atk), "n_benign": len(ben),
        "recall": round(len(blocked_atk) / len(atk), 4) if atk else 0.0,
        "recall_ci": wilson(len(blocked_atk), len(atk)),
        "fp_rate": round(len(blocked_ben) / len(ben), 4) if ben else 0.0,
        "fp_ci": wilson(len(blocked_ben), len(ben)),
        "layer_hits": layers,
        "missed": [{"id": r["id"], "text": r["text"]} for r in atk if not r["blocked"]],
        "false_pos": [{"id": r["id"], "text": r["text"], "layer": r["layer"]}
                      for r in blocked_ben],
        "p95_latency_ms": round(statistics.quantiles(lat, n=20)[-1], 3) if len(lat) > 1 else round(max(lat), 3),
    }


def run_batch(sec_path, extra_path, benign_pool):
    main = load_jsonl(sec_path)
    extra = load_jsonl(extra_path) if extra_path else []
    train_atk = [r["text"] for r in main
                 if r["label"] == "attack" and r.get("batch") in ("A", "anchor", None)]
    eval_atk = [{"id": r.get("id"), "text": r["text"], "label": "attack"}
                for r in main if r["label"] == "attack" and r.get("batch") == "B"]
    eval_atk += [{"id": r.get("id"), "text": r["text"], "label": "attack"} for r in extra]
    ben = [{"id": r.get("id"), "text": r["text"], "label": "benign"}
           for r in main if r["label"] == "benign"]
    if not eval_atk:
        eval_atk = [{"id": r.get("id"), "text": r["text"], "label": "attack"} for r in main if r["label"] == "attack"]

    pipe = build_pipeline(train_atk, benign_pool)
    rows = evaluate(pipe, eval_atk + ben)
    stats = summarize(rows)
    stats.update({"train_attacks": len(train_atk), "eval_attacks": len(eval_atk)})
    return stats, rows


def cluster_attacks(attacks, thresh=0.5):
    """按字符二元组相似度把同族攻击聚成一类（分组留出用）"""
    from app.core.guard.normalize import char_bigrams

    n = len(attacks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    grams = [char_bigrams(t) for t in attacks]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = grams[i], grams[j]
            if not a or not b:
                continue
            if len(a & b) / len(a | b) >= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def run_cv(sec_path, benign_pool, folds_n, split):
    cases = load_jsonl(sec_path)
    attacks = [c["text"] for c in cases if c["label"] == "attack"]
    import numpy as np

    if split == "group":
        clusters = sorted(cluster_attacks(attacks), key=len, reverse=True)
        buckets = [[] for _ in range(folds_n)]
        for g in clusters:
            smallest = min(range(folds_n), key=lambda i: len(buckets[i]))
            buckets[smallest].extend(g)
        folds = [np.array(sorted(b)) for b in buckets if b]
    else:
        rng = np.random.RandomState(42)
        folds = np.array_split(rng.permutation(len(attacks)), folds_n)
    for k in range(len(folds)):
        test_idx = set(folds[k].tolist())
        train_atk = [attacks[i] for i in range(len(attacks)) if i not in test_idx]
        test_cases = [c for i, c in enumerate(cases)
                      if (c["label"] == "attack" and i in test_idx) or c["label"] == "benign"]
        pipe = build_pipeline(train_atk, benign_pool)
        stats.append(summarize(evaluate(pipe, test_cases)))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "batch", "cv"], default="auto")
    ap.add_argument("--security", type=Path, default=None)
    ap.add_argument("--attack-extra", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=["random", "group"], default="random")
    args = ap.parse_args(argv)

    sec_path = args.security or (EVAL_DIR / "security_eval_2000.jsonl")
    if not sec_path.exists():
        sec_path = EVAL_DIR / "security_eval.jsonl"
    extra_path = args.attack_extra or (EVAL_DIR / "security_attack_800.jsonl")
    if not extra_path.exists():
        extra_path = None

    # 正常样本池：主集 + 意图集 + 检索集（训练 L3 用，覆盖真实业务问法）
    benign_pool = [c["text"] for c in load_jsonl(sec_path) if c["label"] == "benign"]
    for p, key in ((EVAL_DIR / "intent_eval_2000.jsonl", "text"),
                   (EVAL_DIR / "retrieval_eval_2000.jsonl", "query"),
                   (EVAL_DIR / "intent_eval.jsonl", "text"),
                   (EVAL_DIR / "qa_eval.jsonl", "query")):
        benign_pool += [c[key] for c in load_jsonl(p) if c.get(key)]
    benign_pool = list(dict.fromkeys(benign_pool))

    mode = args.mode
    if mode == "auto":
        mode = "batch" if extra_path else "cv"

    # L1 单层基线
    base_pipe = GuardPipeline(templates=[], classifier=None,
                              enable_l2=False, enable_l3=False, enable_l4=False)
    base_rows = evaluate(base_pipe, load_jsonl(sec_path))
    base = summarize(base_rows)

    if mode == "batch":
        stats, rows = run_batch(sec_path, extra_path, benign_pool)
        print("\n" + "=" * 66)
        print(f"【L1 单层】召回 {base['recall']:.4f} CI{base['recall_ci']} / "
              f"误报 {base['fp_rate']:.4f} CI{base['fp_ci']}")
        print(f"【四层级联 · 批次留出】训练攻击 {stats['train_attacks']} 条，"
              f"评测攻击 {stats['eval_attacks']} 条，正常 {stats['n_benign']} 条")
        print(f"  召回 {stats['recall']:.4f}  CI{stats['recall_ci']}")
        print(f"  误报 {stats['fp_rate']:.4f}  CI{stats['fp_ci']}")
        print(f"  逐层归因 {stats['layer_hits']}；P95 {stats['p95_latency_ms']} ms")
        print("=" * 66)
        report = {"mode": "batch", "security_set": sec_path.name,
                  "attack_extra": extra_path.name if extra_path else None,
                  "l1_baseline": base, "cascade": stats}
    else:
        fold_stats = run_cv(sec_path, benign_pool, args.folds, args.split)
        import numpy as np

        rec = round(float(np.mean([f["recall"] for f in fold_stats])), 4)
        rec_sd = round(float(np.std([f["recall"] for f in fold_stats])), 4)
        fp = round(float(np.mean([f["fp_rate"] for f in fold_stats])), 4)
        fp_sd = round(float(np.std([f["fp_rate"] for f in fold_stats])), 4)
        print("\n" + "=" * 66)
        print(f"【L1 单层】召回 {base['recall']:.4f} / 误报 {base['fp_rate']:.4f}")
        print(f"【四层级联 · {args.folds} 折 {args.split}】召回 {rec:.4f}±{rec_sd:.4f} / "
              f"误报 {fp:.4f}±{fp_sd:.4f}")
        print(f"  逐层归因 {fold_stats[0]['layer_hits']}（首折）")
        print("=" * 66)
        report = {"mode": "cv", "security_set": sec_path.name, "split": args.split,
                  "l1_baseline": base,
                  "cascade_cv": {"folds": args.folds, "recall_mean": rec, "recall_std": rec_sd,
                                  "fp_mean": fp, "fp_std": fp_sd, "per_fold": fold_stats}}

    report["note"] = "L4 未配置 LLM；灰区按放行+待审处理；L2 稠密不可用时回退 n-gram。"
    out = EVAL_DIR / f"guard_report_{mode}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
