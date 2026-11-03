"""修复意图集 unknown 类的标签污染

问题：用"改写种子"生成 unknown 样本时，模型会把"周末去哪玩"改写成
"图书馆周末开放吗"——句子还是那句话的壳，但意图已经变成 policy_faq，
标签却继承为 unknown，直接把路由评测的准确率打下来（实测 unknown 标签污染
让 accuracy 从 0.83 掉到 0.61）。

修法（可机器验证）：
1. 删掉所有 LLM 生成的 unknown（保留人工 42 条锚点）；
2. 重新生成时强约束"与图书馆完全无关"，并用关键词黑名单过滤；
3. 长度、去重、配额校验后再写回。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gen_eval_sets import (  # noqa: E402
    call_llm, parse_items, norm, add_unique, load_jsonl, write_jsonl,
)

PATH = ROOT / "data" / "eval" / "intent_eval_2000.jsonl"
TARGET = 250
# 与图书馆强相关的词：unknown 样本里出现即视为跑题失败
LIB_WORDS = ["图书馆", "借", "还", "书", "馆藏", "座位", "自习", "开放", "闭馆",
             "办证", "借书证", "续借", "超期", "罚款", "知网", "万方", "VPN",
             "论文", "查重", "打印", "研讨", "楼层", "服务台", "读者", "馆员"]


def main():
    rows = load_jsonl(PATH)
    kept = [r for r in rows if not (r["intent"] == "unknown" and r.get("source") == "llm")]
    removed = len(rows) - len(kept)
    pool = {norm(r["text"]) for r in kept}

    off_topic = ["周末去哪玩", "推荐一部电影", "今天天气怎么样", "怎么减肥", "考研还是就业",
                 "帮我写首诗", "电脑蓝屏了怎么办", "附近有什么好吃的", "怎么学英语",
                 "推荐一款手机", "心情不好怎么办", "怎么和室友相处", "球赛比分",
                 "怎么准备面试", "有什么好看的剧", "怎么养成早起习惯"]
    random.seed(7)
    made = []
    while len(made) + sum(1 for r in kept if r["intent"] == "unknown") < TARGET:
        batch = min(15, TARGET - sum(1 for r in kept if r["intent"] == "unknown") - len(made))
        prompt = (
            f"生成 {batch} 条中文用户随口说的话，要求**与图书馆完全无关**（生活、学习、情绪、娱乐、闲聊均可）。\n"
            "严禁出现这些词：图书馆、借书、还书、座位、自习室、开放时间、知网、论文、打印、馆员。\n"
            "6~35 字，像真实聊天，不要编号，不要问图书馆相关的事。\n"
            '只输出 JSON 数组：[{"text":"..."}]'
        )
        for item in parse_items(call_llm(prompt, max_tokens=900)):
            text = str(item.get("text", "")).strip()
            if not (4 <= len(text) <= 45):
                continue
            if any(w in text for w in LIB_WORDS):
                continue
            if add_unique(text, pool):
                made.append({"text": text, "intent": "unknown", "source": "llm", "batch": "B"})
    out = kept + made
    write_jsonl(PATH, out)
    counts = {}
    for r in out:
        counts[r["intent"]] = counts.get(r["intent"], 0) + 1
    print(f"删除污染样本 {removed} 条，补齐 {len(made)} 条；分布 {counts}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
