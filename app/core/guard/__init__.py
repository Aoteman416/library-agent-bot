"""四层 Prompt 注入安全守卫（Module 3.1 的工程实现）

    L1 正则分类过滤  →  L2 语义向量检索  →  L3 轻量分类器  →  L4 大模型复核
    0.0x ms             几 ms (CPU)          亚毫秒            1~3 s（仅极小灰区）

对外只用三个东西：
- get_pipeline()  ：进程级单例
- pipeline.evaluate(text) → 统一裁决 dict（blocked/layer/category/confidence/...）
- 各层可独立开关（GUARD_ENABLE_L2/L3/L4），资产缺失自动跳过
"""
from .pipeline import GuardPipeline, get_pipeline

__all__ = ["GuardPipeline", "get_pipeline"]
