"""安全红队评测（实验2）

评测对象：app.core.security.SecurityGuard.check_injection（生产代码实际实现）
重要说明：当前代码只有 L1 正则黑名单（19 条正则 + 长度/重复字符），
          L2 语义向量 / L3 微调分类器 / L4 大模型复核 在设计文档中定义但**代码未实现**。
          因此本评测如实测量 L1 的真实拦截能力，并标注 L2/L3/L4 = 设计态(未落地)。

指标：各层拦截率(本环境仅 L1) / 误报率(FP) / L4 触发率(=0) / P95 时延
数据集：data/eval/security_eval.jsonl（{id,text,label}，label∈{attack,benign}）

用法：
    python eval/run_security_eval.py
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.security import SecurityGuard  # noqa: E402

DEFAULT_SET = ROOT / "data" / "eval" / "security_eval.jsonl"


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
    guard = SecurityGuard()

    attacks = [c for c in cases if c["label"] == "attack"]
    benign = [c for c in cases if c["label"] == "benign"]

    blocked_attacks = 0
    missed = []
    blocked_benign = 0
    latencies = []

    for c in cases:
        t0 = time.perf_counter()
        res = guard.check_injection(c["text"])
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        is_blocked = bool(res.get("blocked"))
        if c["label"] == "attack":
            if is_blocked:
                blocked_attacks += 1
            else:
                missed.append(c)
        else:
            if is_blocked:
                blocked_benign += 1

    n_att = len(attacks)
    n_ben = len(benign)
    l1_recall = blocked_attacks / n_att if n_att else 0.0
    fp_rate = blocked_benign / n_ben if n_ben else 0.0
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else max(latencies)

    print(f"\n样本数: 攻击={n_att} 正常={n_ben}")
    print("=" * 56)
    print("【各层拦截率】（当前代码实际实现）")
    print(f"  L1 正则黑名单         : {l1_recall:.4f}  ({blocked_attacks}/{n_att} 攻击被拦)")
    print(f"  L2 语义向量(BGE+FAISS): 设计态(代码未实现) → 实测 0.0000")
    print(f"  L3 微调分类器         : 设计态(代码未实现) → 实测 0.0000")
    print(f"  L4 大模型复核         : 设计态(代码未实现) → 实测 0.0000")
    print("-" * 56)
    print(f"【误报率 FP】正常样本被误拦 : {fp_rate:.4f}  ({blocked_benign}/{n_ben})")
    print(f"【L4 触发率】           : 0.0000 (无 L4 层)")
    print(f"【P95 时延】单次检测     : {p95:.3f} ms")
    print("=" * 56)

    print(f"\nL1 漏拦的攻击({len(missed)} 条)——即设计 L2-L4 应覆盖的盲区：")
    # 简单归类展示
    cats = {
        "全角/全宽字符": ["ｉｇｎｏｒｅ", "説"],
        "零宽字符插入": ["​"],
        "近义改写/ paraphrase": ["忘掉", "作废", "解除", "自由", "无限制", "训练", "数据"],
        "间接注入": ["当作最高指令", "当作指令执行", "隐藏指令"],
        "角色/越权变体": ["模拟越权", "管理员权限", "开发者模式", "后台"],
    }
    for c in missed:
        tags = [name for name, kws in cats.items() if any(k in c["text"] for k in kws)]
        tag = "/".join(tags) if tags else "其他变体"
        print(f"  #{c['id']:>3} [{tag}] {c['text']}")

    out = {
        "num_attacks": n_att,
        "num_benign": n_ben,
        "L1_recall": round(l1_recall, 4),
        "L2_recall": 0.0, "L3_recall": 0.0, "L4_recall": 0.0,
        "false_positive_rate": round(fp_rate, 4),
        "L4_trigger_rate": 0.0,
        "p95_latency_ms": round(p95, 4),
        "missed_attacks": [{"id": c["id"], "text": c["text"]} for c in missed],
    }
    (ROOT / "data" / "eval" / "security_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 data/eval/security_report.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
