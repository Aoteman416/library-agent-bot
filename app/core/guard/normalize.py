"""安全守卫预处理：把一条输入展开为多个"分析视图"，对抗编码/插入/全角等混淆。

设计动机（Module 3.1.1）：
攻击者绕过正则最常用的三招是「全角替换」「零宽字符插入」「字符间插分隔符」。
如果直接拿原文去跑正则，规则永远追不上变体；正确做法是先归一化，
再让同一批规则并行扫描多个视图——任一视图命中即算命中，不牺牲速度。

视图说明：
- raw      : 原始输入（保语义完整，传给 L2 用原文，避免归一化丢失语义）
- nfkc     : Unicode NFKC 归一（全角→半角、兼容字符统一）+ 去零宽/不可见字符
- deobf    : nfkc 基础上，去除中文字符之间的分隔符（"忽 略 之 前"→"忽略之前"）
- compact  : 去掉所有空白与标点，只留字/数字/字母（防标点轰炸）
"""
import re
import unicodedata
from typing import Dict

# 零宽 / 方向控制 / BOM 等不可见字符
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u180e\ufeff]")

# 夹在中文之间的分隔符（空格、点、横线、星号、竖线等）
_CJK_SEP = re.compile(r"(?<=[\u4e00-\u9fff])[\s\.\-_*·|/\\~,;:：；，。、'\"()\[\]{}<>]+(?=[\u4e00-\u9fff])")

_NON_WORD = re.compile(r"[^\u4e00-\u9fff0-9a-zA-Z]+")
_MULTI_SPACE = re.compile(r"\s+")


def strip_invisible(text: str) -> str:
    """去掉零宽与不可见控制字符（攻击者用来打断关键词的最廉价手段）"""
    return _INVISIBLE.sub("", text)


def views(text: str) -> Dict[str, str]:
    """生成多视图：raw / nfkc / deobf / despace / compact"""
    raw = text or ""
    nfkc = strip_invisible(unicodedata.normalize("NFKC", raw))
    deobf = _CJK_SEP.sub("", nfkc)
    deobf = _MULTI_SPACE.sub(" ", deobf).strip()
    # 去空白视图：对抗 "i g n o r e  t h e  r u l e s" 这种字母间插空格
    despace = _MULTI_SPACE.sub("", nfkc)
    compact = _NON_WORD.sub("", nfkc.lower())
    return {"raw": raw, "nfkc": nfkc, "deobf": deobf,
            "despace": despace, "compact": compact}


def char_bigrams(text: str) -> set:
    """字符二元组集合：L2 的关键词重叠辅助信号、L3 的兜底特征都用它"""
    compact = _NON_WORD.sub("", (text or "").lower())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def overlap_ratio(a: str, b: str) -> float:
    """两段文本的字符二元组 Jaccard 相似度（0~1）"""
    sa, sb = char_bigrams(a), char_bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def anomaly_score(text: str) -> float:
    """长度/重复字符异常度（0~1），作为 L2 的辅助信号，不单独拦截"""
    if not text:
        return 0.0
    score = 0.0
    if len(text) > 1000:
        score += min((len(text) - 1000) / 2000.0, 0.5)
    if re.search(r"(.)\1{8,}", text):
        score += 0.3
    return min(score, 1.0)
