"""把叙述类文档里的旧口径数字统一到 2000 条级最新结果

只改"当前口径"文档（故事版/话术/简历/经验映射/旧版归档），
不改已带历史对照表的《实验评测结论》和事实库（那里保留 200/300 锚点是刻意的）。

用法：
    python scripts/unify_numbers.py --dry-run
    python scripts/unify_numbers.py
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    "interview/图书馆故事版.md",
    "interview/图书馆背诵话术.md",
    "interview/图书馆简历内容.md",
    "interview/图书馆经验➡项目映射.md",
    "interview/project-profile.md",
    "docs/关于《图书馆智能客服》的背诵版.md",
    "docs/关于《图书馆智能客服》的故事版.md",
    "docs/关于《图书馆智能客服》的简历内容.md",
    "docs/关于《图书馆智能客服》的经验-项目映射.md",
    "docs/关于《图书馆智能客服》的项目事实库.md",
]

# 顺序敏感：先长后短，避免 "0.7071±0.0827" 被拆掉
REPLACEMENTS = [
    ("0.7071±0.0827", "0.4800（CI 0.4456–0.5146）"),
    ("0.7071 ± 0.0827", "0.4800（CI 0.4456–0.5146）"),
    ("0.7071", "0.4800"),
    ("0.6357", "0.5100"),
    ("89/140", "102/200"),
    ("0.0667", "0.0089"),
    ("6.67%", "0.89%"),
    ("1.1ms", "1.887ms"),
    ("0.995", "0.9892"),
    ("0.9043", "0.8998"),
    ("1588", "1947"),
    ("分组留出", "批次留出"),
    ("重复续借约 5%", "重复续借 11.33%"),
    ("约 5% 重复续借", "11.33% 重复续借"),
    ("5% 丢失更新", "11.33% 丢失更新"),
    ("80 轮×20 线程", "300 轮×50 线程"),
    ("80 轮 × 20 线程", "300 轮 × 50 线程"),
    ("80 轮*20 线程", "300 轮*50 线程"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"跳过（不存在）：{rel}")
            continue
        text = p.read_text(encoding="utf-8")
        orig = text
        hits = {}
        for old, new in REPLACEMENTS:
            c = text.count(old)
            if c:
                hits[old] = c
                text = text.replace(old, new)
        if text != orig:
            total += sum(hits.values())
            print(f"{rel}: {sum(hits.values())} 处  {hits}")
            if not args.dry_run:
                p.write_text(text, encoding="utf-8")
    print(f"\n合计替换 {total} 处" + ("（dry-run，未写盘）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
