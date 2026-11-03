# RAG 检索质量评测报告

| 配置 | Top-K | Recall | Precision | MRR | HitRate | KeywordHit | nDCG |
|---|---|---|---|---|---|---|
| ngram+memory+bm25(recall) | 1 | 0.805 | 0.805 | 0.805 | 0.805 | 0.845 | 0.805 |
| ngram+memory+bm25(recall) | 3 | 0.92 | 0.3067 | 0.8542 | 0.92 | 0.93 | 0.871 |
| ngram+memory+bm25(recall) | 5 | 0.995 | 0.199 | 0.8719 | 0.995 | 0.995 | 0.9024 |

## ngram+memory+bm25(recall) 未达标样本（1）

| ID | 问题 | 问题类型 | Top1 实际命中 | 期望 |
|---|---|---|---|---|
| 73 | 非本校学生能进图书馆吗 | 召回失败 | 开放时间与办证.md#办证方法 | 开放时间与办证.md#校外访客 |
