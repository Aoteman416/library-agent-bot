"""检索评测（实验3）——多路召回对比

对比变体（对应面试故事 S-P1-003）：
  1) 纯稀疏 ngram-TF-IDF        （零依赖基线）
  2) BM25 融合（ngram + BM25）  （关键词一路）
  3) 纯稠密 BGE-M3              （语义一路，无 BM25）
  4) 混合（BGE-M3 + BM25 融合） （双路）
  5) 重排（混合 + CrossEncoder） （双路 + 精排）

指标：Recall@5 / MRR / nDCG@5（nDCG 由 run_eval 计算，二值相关性）
数据集：data/eval/qa_eval.jsonl（48 条人工标注）

用法：
    python eval/run_retrieval_eval.py
"""
import json
import os
import sys
from pathlib import Path

# 在导入 app 配置前设定模型，控制下载体积（bge-m3 2.2GB→bge-small 100MB）
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_eval import build_retriever, evaluate, load_eval_set  # noqa: E402

OUT = ROOT / "data" / "eval"

# (label, embedding_mode, enable_bm25, use_rerank)
VARIANTS = [
    ("纯稀疏-ngramTFIDF", "ngram", False, False),
    ("BM25融合-ngram+BM25", "ngram", True, False),
    ("纯稠密-BGE", "external", False, False),
    ("混合-BGE+BM25", "external", True, False),
    ("重排-混合+CrossEncoder", "external", True, True),
]


def _hf_reachable() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data" / "eval" / "qa_eval.jsonl")
    args = ap.parse_args()
    cases = load_eval_set(args.eval_set)
    hf_ok = _hf_reachable()
    print(f"评测集：{args.eval_set.name}（{len(cases)} 条）；HuggingFace 可达：{hf_ok}\n")
    summaries = []
    for label, mode, bm25, rerank in VARIANTS:
        # 稠密/重排变体需要下载 BGE 模型；本环境若 HF 不可达则明确跳过，避免无意义超时
        if mode == "external" and not hf_ok:
            print(f"[ENV-BLOCKED] {label}：HuggingFace 不可达，无法下载 BGE 模型。"
                  f"代码路径已就绪（--mode external --rerank），在可访问模型的环境运行即得。")
            summaries.append({
                "label": label, "metrics": {}, "failures": [],
                "env_blocked": True,
                "note": "需 BAAI/bge-m3 或 bge-small + bge-reranker-v2-m3，本环境 HuggingFace 不可达",
            })
            continue
        try:
            print(f"构建并评测：{label} ...")
            retriever, chunks = build_retriever(
                embedding_mode=mode,
                vector_store_mode="memory",
                collection_name=f"retrieval_eval_{label}",
                enable_bm25=bm25,
                fusion="rrf",
            )
            s = evaluate(
                retriever, chunks, cases,
                top_k_values=(1, 3, 5),
                use_rerank=rerank, label=label,
            )
            summaries.append(s)
            m5 = s["metrics"][5]
            print(f"  → Recall@5={m5['recall@5']}  MRR={m5['mrr']}  nDCG@5={m5['ndcg@5']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 跳过 {label}：{type(exc).__name__} {exc}")

    # 对比表（聚焦 K=5）
    print("\n=== 变体对比（K=5）===")
    print(f"{'变体':<26}{'Recall@5':>10}{'MRR':>9}{'nDCG@5':>9}")
    print("-" * 56)
    for s in summaries:
        if s.get("env_blocked"):
            print(f"{s['label']:<24}{'ENV-BLOCKED':>10}")
            continue
        m = s["metrics"][5]
        print(f"{s['label']:<24}{m['recall@5']:>10}{m['mrr']:>9}{m['ndcg@5']:>9}")

    (OUT / "retrieval_report.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 data/eval/retrieval_report.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
