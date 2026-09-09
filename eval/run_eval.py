"""RAG 检索质量评测

评测集：data/eval/qa_eval.jsonl（人工标注，每条标注了期望命中的知识块与必含关键词）

指标口径（面试可讲）：
- Recall@K    = |Top-K ∩ 相关分块| / |相关分块总数|        衡量召全能力
- Precision@K = |Top-K ∩ 相关分块| / K                     衡量排得准不准
- MRR         = 1 / 第一个相关结果的排名                    衡量首条命中能力
- HitRate@K   = Top-K 中是否出现相关分块                    衡量可用率
- KeywordHit@K= Top-K 拼接文本是否覆盖全部必含关键词        衡量「能否直接回答」

「相关」的判定 = 分块的 source 与 category 元数据同时命中标注值，
因此指标衡量的最小粒度是「知识条目」而非「整篇文档」。

用法：
    python eval/run_eval.py                       # 默认配置（稀疏 + 内存）评测
    python eval/run_eval.py --top-k 1 --top-k 3   # 指定多个 K
    python eval/run_eval.py --no-rerank           # 只测向量召回，跳过精排
    python eval/run_eval.py --compare             # 稀疏 n-gram vs 稠密 BGE 对比
    python eval/run_eval.py --mode external       # 只用稠密 BGE 评测
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# 保证可以直接 `python eval/run_eval.py` 运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.chunker import TextChunker  # noqa: E402
from app.rag.embeddings import create_embedder  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.vector_store import create_store  # noqa: E402
from app.services.knowledge_service import KnowledgeService  # noqa: E402

DEFAULT_EVAL_SET = ROOT / "data" / "eval" / "qa_eval.jsonl"
DEFAULT_REPORT_DIR = ROOT / "data" / "eval"

# 记忆/ngram 是稀疏向量，只能配内存库；稠密模型两者皆可
STORE_COMPAT = {"ngram": "memory", "external": "memory"}


# ==================== 评测集 ====================
@dataclass
class EvalCase:
    """一条人工标注的评测样本"""

    id: int
    query: str
    relevant_source: str
    relevant_category: str = ""
    must_contain: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalCase":
        return cls(
            id=int(data["id"]),
            query=data["query"],
            relevant_source=data["relevant_source"],
            relevant_category=data.get("relevant_category", ""),
            must_contain=list(data.get("must_contain", [])),
        )


def load_eval_set(path: Path = DEFAULT_EVAL_SET) -> List[EvalCase]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


# ==================== 检索器构建 ====================
def build_retriever(
    embedding_mode: str = "ngram",
    vector_store_mode: str = "memory",
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    collection_name: Optional[str] = None,
):
    """构建一个独立的检索器用于评测（不污染服务运行时的单例索引）"""
    knowledge = KnowledgeService()
    knowledge.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    knowledge.load_directory()

    embedder = create_embedder(embedding_mode)
    embedder.fit(knowledge.corpus)

    # 评测使用独立的 Chroma 集合，避免覆盖服务索引
    store_kwargs = {}
    if collection_name:
        store_kwargs["collection_name"] = collection_name
    store = create_store(vector_store_mode, **store_kwargs)

    chunks = knowledge.chunks
    store.clear()
    if chunks:
        vectors = embedder.embed_batch([c.content for c in chunks])
        if hasattr(store, "add_batch"):
            store.add_batch(chunks, vectors)
        else:
            for chunk, vector in zip(chunks, vectors):
                store.add(chunk, vector)

    return Retriever(embedder, store), chunks


# ==================== 指标计算 ====================
def relevant_chunk_ids(case: EvalCase, chunks) -> set:
    """该 query 对应的全部相关分块（知识条目粒度）"""
    return {
        c.chunk_id
        for c in chunks
        if c.metadata.get("source") == case.relevant_source
        and (
            not case.relevant_category
            or c.metadata.get("category") == case.relevant_category
        )
    }


@dataclass
class CaseResult:
    case_id: int
    query: str
    recall: float
    precision: float
    reciprocal_rank: float
    hit: int
    keyword_hit: int
    top1_source: str


def score_results(
    case: EvalCase,
    results: Sequence[Dict],
    relevant: set,
    top_k: int,
) -> CaseResult:
    """对一组检索结果计算单条样本的指标"""
    retrieved_ids = [r["chunk"].chunk_id for r in results]
    hits = [cid for cid in retrieved_ids if cid in relevant]

    recall = len(hits) / len(relevant) if relevant else 0.0
    precision = len(hits) / top_k if top_k else 0.0

    reciprocal_rank = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            reciprocal_rank = 1.0 / rank
            break

    concat = "".join(r["chunk"].content for r in results)
    keyword_hit = (
        int(all(kw in concat for kw in case.must_contain)) if case.must_contain else -1
    )

    top1_source = ""
    if results:
        meta = results[0]["chunk"].metadata
        top1_source = f"{meta.get('source', '?')}#{meta.get('category', '?')}"

    return CaseResult(
        case_id=case.id,
        query=case.query,
        recall=recall,
        precision=precision,
        reciprocal_rank=reciprocal_rank,
        hit=int(bool(hits)),
        keyword_hit=keyword_hit,
        top1_source=top1_source,
    )


def evaluate(
    retriever: Retriever,
    chunks,
    cases: Sequence[EvalCase],
    top_k_values: Sequence[int] = (1, 3, 5),
    use_rerank: bool = True,
    label: str = "default",
) -> Dict:
    """在多个 K 上评测，返回聚合结果

    每个样本只检索一次（取最大 K），其余 K 直接截断前缀，
    这样 K=1/3/5 的指标来自同一次排序，可比性更强。
    """
    max_k = max(top_k_values)
    raw: Dict[int, List[CaseResult]] = {k: [] for k in top_k_values}

    for case in cases:
        if use_rerank:
            results = retriever.retrieve(case.query, top_k=max_k, rerank_top_n=max_k)
        else:
            results = retriever.recall(case.query, top_k=max_k)

        relevant = relevant_chunk_ids(case, chunks)
        for k in top_k_values:
            raw[k].append(score_results(case, results[:k], relevant, k))

    summary = {"label": label, "num_cases": len(cases), "metrics": {}, "failures": []}
    for k in top_k_values:
        rs = raw[k]
        n = len(rs) or 1
        kw_results = [r.keyword_hit for r in rs if r.keyword_hit >= 0]
        summary["metrics"][k] = {
            f"recall@{k}": round(sum(r.recall for r in rs) / n, 4),
            f"precision@{k}": round(sum(r.precision for r in rs) / n, 4),
            f"hit_rate@{k}": round(sum(r.hit for r in rs) / n, 4),
            f"keyword_hit_rate@{k}": (
                round(sum(kw_results) / len(kw_results), 4) if kw_results else None
            ),
            "mrr": round(sum(r.reciprocal_rank for r in rs) / n, 4),
        }

    # 记录最差 K 下的失败样本，供人工分析与知识补录
    worst_k = max(top_k_values)
    for case, result in zip(cases, raw[worst_k]):
        if not result.hit or result.keyword_hit == 0:
            summary["failures"].append(
                {
                    "id": case.id,
                    "query": case.query,
                    "hit": bool(result.hit),
                    "keyword_hit": bool(result.keyword_hit),
                    "top1": result.top1_source,
                    "expected": f"{case.relevant_source}#{case.relevant_category}",
                }
            )
    return summary


# ==================== 报告输出 ====================
def format_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
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


def summary_rows(summaries: Sequence[Dict]) -> List[List]:
    rows = []
    for s in summaries:
        for k, m in s["metrics"].items():
            rows.append(
                [
                    s["label"],
                    k,
                    m[f"recall@{k}"],
                    m[f"precision@{k}"],
                    m["mrr"],
                    m[f"hit_rate@{k}"],
                    m[f"keyword_hit_rate@{k}"],
                ]
            )
    return rows


def print_report(summaries: Sequence[Dict]) -> None:
    headers = ["配置", "Top-K", "Recall", "Precision", "MRR", "HitRate", "KeywordHit"]
    print()
    print(format_table(headers, summary_rows(summaries)))
    print()
    for s in summaries:
        if s["failures"]:
            print(f"[{s['label']}] 未达标样本 {len(s['failures'])} 条：")
            for f in s["failures"][:10]:
                flag = "召回失败" if not f["hit"] else "关键词缺失"
                print(f"  - #{f['id']} {f['query']}  ({flag}) top1={f['top1']} 期望={f['expected']}")
            print()


def write_report(summaries: Sequence[Dict], output_dir: Path = DEFAULT_REPORT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# RAG 检索质量评测报告", ""]
    lines.append("| 配置 | Top-K | Recall | Precision | MRR | HitRate | KeywordHit |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in summary_rows(summaries):
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    for s in summaries:
        if s["failures"]:
            lines.append(f"## {s['label']} 未达标样本（{len(s['failures'])}）")
            lines.append("")
            lines.append("| ID | 问题 | 问题类型 | Top1 实际命中 | 期望 |")
            lines.append("|---|---|---|---|---|")
            for f in s["failures"]:
                flag = "召回失败" if not f["hit"] else "关键词缺失"
                lines.append(
                    f"| {f['id']} | {f['query']} | {flag} | {f['top1']} | {f['expected']} |"
                )
            lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ==================== 入口 ====================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--mode", default="ngram", choices=["ngram", "external"], help="Embedding 模式"
    )
    parser.add_argument(
        "--store", default="", choices=["", "memory", "chroma"], help="向量库模式"
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[1, 3, 5], help="评测的 K 值（可多个）"
    )
    parser.add_argument("--no-rerank", action="store_true", help="只测向量召回，跳过精排")
    parser.add_argument("--compare", action="store_true", help="对比稀疏与稠密两种检索器")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cases = load_eval_set(args.eval_set)
    print(f"评测集：{args.eval_set.name}，共 {len(cases)} 条标注样本")

    configs = []
    if args.compare:
        configs = [("ngram", "memory"), ("external", args.store or "memory")]
    else:
        configs = [(args.mode, args.store or STORE_COMPAT.get(args.mode, "memory"))]

    summaries = []
    for mode, store_mode in configs:
        label = f"{mode}+{store_mode}" + ("(recall)" if args.no_rerank else "")
        print(f"\n构建检索器：{label} ...")
        try:
            retriever, chunks = build_retriever(
                embedding_mode=mode,
                vector_store_mode=store_mode,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                collection_name=f"rag_eval_{mode}",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 跳过 {label}：{exc}")
            continue
        print(f"  索引就绪：{len(chunks)} 个分块")
        summaries.append(
            evaluate(
                retriever,
                chunks,
                cases,
                top_k_values=args.top_k,
                use_rerank=not args.no_rerank,
                label=label,
            )
        )

    if not summaries:
        print("没有任何可评测的检索器")
        return 1

    print_report(summaries)
    write_report(summaries)
    print(f"报告已写入：{DEFAULT_REPORT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
