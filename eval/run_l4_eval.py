"""L4 大模型复核评测（灰区集）

流程：
1. 用 L1–L3 跑全量安全集（2000 + 800 攻击专项），收集"灰区"样本
   （即 L3 判 review、需要 L4 终裁的样本）；
2. 灰区里的攻击/正常各采样（默认各 120 条），逐条送 L4 复核；
3. 统计：L4 在灰区攻击上的拦截率、在灰区正常上的误判率、超时/解析失败率、
   P50/P95 时延，以及"加上 L4 后整体召回/误报"的变化。

口径：
- 模型：默认免费档 Qwen2.5-7B-Instruct（可用 L4_MODEL 覆盖），不改 .env；
- 超时 3 秒，超时/解析失败按"放行+待审"计（fail-open），不计入拦截；
- 输出 data/eval/l4_report.json。

用法：
    python eval/run_l4_eval.py
    python eval/run_l4_eval.py --per-class 80 --workers 4
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 免费档模型 + 离线向量，避免动 .env、避免拉 BGE
os.environ.setdefault("EMBEDDING_MODE", "ngram")
os.environ.setdefault("GUARD_ENABLE_L4", "0")
os.environ["LLM_MODEL"] = os.getenv("L4_MODEL", "Qwen/Qwen2.5-7B-Instruct")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.guard import assets  # noqa: E402
from app.core.guard.l4_llm import L4LLMJudge  # noqa: E402
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


def build_pipeline(train_attacks, benign):
    templates = assets.build_templates(train_attacks)
    samples = [(t["text"], t["category"]) for t in templates] + [(b, "normal") for b in benign]
    clf = assets.train_l3(samples)
    # 显式关闭 L4，先跑出灰区
    return GuardPipeline(templates=templates, classifier=clf, enable_l4=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120, help="灰区攻击/正常各采样多少条")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    sec = load_jsonl(EVAL_DIR / "security_eval_2000.jsonl")
    extra = load_jsonl(EVAL_DIR / "security_attack_800.jsonl")
    attacks = [r["text"] for r in sec if r["label"] == "attack" and r.get("batch") in ("A", "anchor", None)]
    eval_atk = [r for r in sec if r["label"] == "attack" and r.get("batch") == "B"] + extra
    benign = [r["text"] for r in sec if r["label"] == "benign"]
    pool = benign + [c["text"] for c in load_jsonl(EVAL_DIR / "intent_eval_2000.jsonl") if c.get("text")]
    pool += [c["query"] for c in load_jsonl(EVAL_DIR / "retrieval_eval_2000.jsonl") if c.get("query")]

    pipe = build_pipeline(attacks, list(dict.fromkeys(pool)))
    print(f"资产：训练攻击 {len(attacks)}，评测攻击 {len(eval_atk)}，正常 {len(benign)}")

    def is_gray(res):
        return any(l.get("layer") == "L3" and l.get("action") == "review" for l in res.get("layers", []))

    gray_atk, gray_ben, total_atk_blocked = [], [], 0
    for i, r in enumerate(eval_atk):
        res = pipe.evaluate(r["text"])
        if res["blocked"]:
            total_atk_blocked += 1
        elif is_gray(res):
            gray_atk.append(r["text"])
        if (i + 1) % 200 == 0:
            print(f"  攻击扫描 {i + 1}/{len(eval_atk)}，灰区 {len(gray_atk)}")

    ben_blocked = 0
    for i, text in enumerate(benign):
        res = pipe.evaluate(text)
        if res["blocked"]:
            ben_blocked += 1
        elif is_gray(res):
            gray_ben.append(text)
        if (i + 1) % 400 == 0:
            print(f"  正常扫描 {i + 1}/{len(benign)}，灰区 {len(gray_ben)}")

    print(f"L1–L3 结果：攻击拦截 {total_atk_blocked}/{len(eval_atk)}，"
          f"正常误报 {ben_blocked}/{len(benign)}，灰区 攻击 {len(gray_atk)} / 正常 {len(gray_ben)}")

    sample_atk = gray_atk[:args.per_class]
    sample_ben = gray_ben[:args.per_class]
    judge = L4LLMJudge(timeout=3.0)
    print(f"L4 可用：{judge.enabled}；评测样本 攻击 {len(sample_atk)} / 正常 {len(sample_ben)}")

    def run_one(text):
        t0 = time.perf_counter()
        r = judge.judge(text)
        return {"text": text, "is_attack": bool(r.get("is_attack")), "ok": bool(r.get("ok")),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "reason": r.get("reason", "")}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        atk_res = list(ex.map(run_one, sample_atk))
        ben_res = list(ex.map(run_one, sample_ben))
    wall = time.perf_counter() - t0

    atk_flag = sum(1 for r in atk_res if r["is_attack"])
    ben_flag = sum(1 for r in ben_res if r["is_attack"])
    fail = sum(1 for r in atk_res + ben_res if not r["ok"])
    lat = [r["latency_ms"] for r in atk_res + ben_res]
    p95 = round(statistics.quantiles(lat, n=20)[-1], 1) if len(lat) > 1 else round(max(lat or [0]), 1)

    total_atk = len(eval_atk)
    total_ben = len(benign)
    recall_before = total_atk_blocked / total_atk
    recall_after = (total_atk_blocked + atk_flag) / total_atk
    fp_before = ben_blocked / total_ben
    fp_after = (ben_blocked + ben_flag) / total_ben

    report = {
        "model": os.environ["LLM_MODEL"],
        "timeout_s": 3.0,
        "counts": {"attack_total": total_atk, "benign_total": total_ben,
                   "gray_attack": len(gray_atk), "gray_benign": len(gray_ben),
                   "eval_attack": len(sample_atk), "eval_benign": len(sample_ben)},
        "l4": {"attack_flagged": atk_flag, "benign_flagged": ben_flag,
               "fail_or_timeout": fail, "fail_rate": round(fail / max(len(lat), 1), 4),
               "p95_latency_ms": p95, "wall_seconds": round(wall, 1)},
        "overall": {"recall_before": round(recall_before, 4), "recall_after": round(recall_after, 4),
                    "fp_before": round(fp_before, 4), "fp_after": round(fp_after, 4)},
        "samples": {"attack": atk_res[:10], "benign": ben_res[:10]},
    }
    (EVAL_DIR / "l4_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"灰区样本：攻击 {len(sample_atk)}（L4 判攻击 {atk_flag}，"
          f"拦截率 {atk_flag / max(len(sample_atk), 1):.4f}）")
    print(f"          正常 {len(sample_ben)}（L4 误判 {ben_flag}，"
          f"误判率 {ben_flag / max(len(sample_ben), 1):.4f}）")
    print(f"超时/解析失败 {fail}/{len(lat)}（fail-open 放行待审）；P95 {p95} ms；总耗时 {wall:.1f}s")
    print(f"整体召回：{recall_before:.4f} → {recall_after:.4f}；"
          f"整体误报：{fp_before:.4f} → {fp_after:.4f}")
    print("=" * 62)
    print("报告：data/eval/l4_report.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
