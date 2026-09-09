"""RAG 检索质量回归测试

把 eval/run_eval.py 的评测能力接入 pytest，形成「数据 → 评测 → 优化」的自动化闭环：
- 指标跌破阈值即测试失败，防止改分块策略 / 换 Embedding / 加知识时悄悄劣化
- 同时校验评测集标注本身的自洽性，避免标注漂移后指标失去意义

阈值取自当前基线（稀疏 n-gram）并留出约 3 个点的余量；
若某次改动是有意提升，请同步更新阈值并在 PR 说明中给出新旧对比。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_eval import build_retriever, evaluate, load_eval_set  # noqa: E402

# 基线（48 条标注样本，21 个知识分块）：
#   Recall@1=0.854  Recall@3=0.958  Recall@5=1.000
#   MRR@3=0.896     HitRate@3=0.958  KeywordHit@3=0.958
THRESHOLDS = {
    1: {"recall@1": 0.80, "mrr": 0.80},
    3: {"recall@3": 0.92, "precision@3": 0.28, "mrr": 0.85,
        "hit_rate@3": 0.92, "keyword_hit_rate@3": 0.92},
    5: {"recall@5": 0.98, "hit_rate@5": 0.98, "keyword_hit_rate@5": 0.98},
}


@pytest.fixture(scope="module")
def eval_cases():
    return load_eval_set()


@pytest.fixture(scope="module")
def sparse_bundle():
    """稀疏 n-gram + 内存索引：零依赖，任何环境都能跑"""
    return build_retriever(embedding_mode="ngram", vector_store_mode="memory")


@pytest.fixture(scope="module")
def sparse_summary(eval_cases, sparse_bundle):
    retriever, chunks = sparse_bundle
    return evaluate(
        retriever,
        chunks,
        eval_cases,
        top_k_values=(1, 3, 5),
        label="ngram+memory",
    )


# ---------- 评测集自洽性 ----------
def test_eval_set_not_empty(eval_cases):
    assert len(eval_cases) >= 30, "评测集样本量过小，指标不具备统计意义"


def test_eval_labels_exist_in_knowledge_base(eval_cases, sparse_bundle):
    """标注的 source#category 必须真实存在，否则知识库改版后标注会静默失效"""
    _, chunks = sparse_bundle
    valid = {(c.metadata.get("source"), c.metadata.get("category")) for c in chunks}
    missing = [
        (c.id, f"{c.relevant_source}#{c.relevant_category}")
        for c in eval_cases
        if (c.relevant_source, c.relevant_category) not in valid
    ]
    assert not missing, f"标注指向了不存在的知识条目：{missing}"


def test_eval_keywords_exist_in_target_chunk(eval_cases, sparse_bundle):
    """必含关键词必须出现在目标分块里，否则该样本的 KeywordHit 永远不可能达标"""
    _, chunks = sparse_bundle
    problems = []
    for case in eval_cases:
        targets = [
            c
            for c in chunks
            if c.metadata.get("source") == case.relevant_source
            and c.metadata.get("category") == case.relevant_category
        ]
        text = "".join(t.content for t in targets)
        for kw in case.must_contain:
            if kw not in text:
                problems.append((case.id, kw))
    assert not problems, f"标注的必含关键词不在目标分块中：{problems}"


# ---------- 检索质量回归 ----------
@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_sparse_retrieval_quality(sparse_summary, top_k):
    metrics = sparse_summary["metrics"][top_k]
    for name, floor in THRESHOLDS[top_k].items():
        assert metrics[name] >= floor, (
            f"Top-{top_k} 的 {name} 跌破阈值：{metrics[name]} < {floor}"
        )


def test_retrieval_rerank_improves_top1(eval_cases, sparse_bundle):
    """精排应当不劣于纯向量召回（融合词项覆盖后 Top-1 准确率至少持平）"""
    retriever, chunks = sparse_bundle
    reranked = evaluate(retriever, chunks, eval_cases, top_k_values=(1,), label="rerank")
    recalled = evaluate(
        retriever, chunks, eval_cases, top_k_values=(1,), use_rerank=False, label="recall"
    )
    assert reranked["metrics"][1]["recall@1"] >= recalled["metrics"][1]["recall@1"]


def test_report_is_generated(sparse_summary):
    """评测报告落盘，供人工复盘与知识补录"""
    report = ROOT / "data" / "eval" / "report.json"
    assert report.exists(), "请先运行 python eval/run_eval.py 生成报告"
    assert sparse_summary["num_cases"] == len(load_eval_set())


# ---------- 稠密检索（依赖缺失时自动跳过）----------
def test_dense_retrieval_quality(eval_cases):
    pytest.importorskip(
        "sentence_transformers", reason="未安装 sentence-transformers，跳过稠密检索评测"
    )
    retriever, chunks = build_retriever(embedding_mode="external", vector_store_mode="memory")
    summary = evaluate(retriever, chunks, eval_cases, top_k_values=(1, 3), label="external")
    # 稠密模型在小知识库上应至少达到与稀疏检索同级别的首条命中能力
    assert summary["metrics"][3]["hit_rate@3"] >= 0.90, (
        f"稠密检索 HitRate@3 偏低：{summary['metrics'][3]['hit_rate@3']}"
    )
