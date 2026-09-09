"""自然语言解析工具：时间、日期、实体"""
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

_CN_NUM_MAP = str.maketrans("零一二三四五六七八九两", "01234567892")

_DATE_KEYWORDS = [
    ("今天", 0), ("今日", 0),
    ("明天", 1), ("明日", 1),
    ("后天", 2), ("大后天", 3),
]

_AREA_KEYWORDS = {
    "3楼自习室": ["3楼", "三楼", "3楼自习", "自习室"],
    "4楼自习室": ["4楼", "四楼", "考研自习室"],
    "5楼研习间": ["5楼", "五楼", "研习间"],
    "2楼文学区": ["2楼", "二楼"],
    "1楼总服务台": ["1楼", "一楼", "总服务台"],
}


def parse_time(text: str) -> Optional[Tuple[str, str]]:
    """
    解析自然语言时间为 (time_str "HH:MM", period)
    
    支持：
    - 下午3点 / 上午10点半 / 晚上8点
    - 15:30 / 14点30
    - 3点（默认下午）
    - "半" = 30分
    """
    # 处理"半"字为 "30"
    text = text.replace("半", "30分")

    # 模式1: 上午/下午/晚上 + 时间
    #   组1=period, 组2=hour, 组3=minute(可选)
    m = re.search(r"(早|上|下|晚)午?\s*(\d{1,2})[点时:](\d{1,2})?分?", text)
    if m:
        period_cn = m.group(1)
        hour = int(m.group(2))
        minute = int(m.group(3)) if m.group(3) else 0

        if period_cn in ("下", "晚"):
            if hour < 12:
                hour += 12
        elif period_cn == "早":
            pass
        elif period_cn == "上":
            if hour == 12:
                hour = 0

        period = "pm" if period_cn in ("下", "晚") else "am"
        return (f"{hour:02d}:{minute:02d}", period)

    # 模式2: 24小时制时间 (15:30 / 14点30)
    m = re.search(r"(\d{1,2})[点时:](\d{1,2})分?", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        if 0 <= hour <= 23:
            period = "pm" if hour >= 12 else "am"
            return (f"{hour:02d}:{minute:02d}", period)

    # 模式3: 只有几点 (默认下午)
    m = re.search(r"(\d{1,2})[点时]", text)
    if m:
        hour = int(m.group(1))
        if hour <= 12:
            return (f"{hour + 12:02d}:00", "pm")
        return (f"{hour:02d}:00", "pm")

    return None


def parse_date(text: str) -> Optional[str]:
    """解析日期为 YYYY-MM-DD"""
    # YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # MM-DD
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = datetime.now().year
        try:
            target = datetime(year, month, day)
            if target < datetime.now():
                year += 1
            return f"{year}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    # 相对日期
    for keyword, delta in _DATE_KEYWORDS:
        if keyword in text:
            target = datetime.now() + timedelta(days=delta)
            return target.strftime("%Y-%m-%d")

    return None


def parse_area(text: str) -> Optional[str]:
    """解析区域"""
    for area, keywords in _AREA_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return area
    return None


def extract_entities(text: str) -> dict:
    """从自然语言中抽取所有实体"""
    date = parse_date(text)
    time_result = parse_time(text)
    area = parse_area(text)

    start_time = None
    end_time = None

    if time_result:
        start_time = time_result[0]
        # 检测时间段范围：如 "下午2-4点" 或 "2点到4点"
        range_match = re.search(
            r"(?:(早|上|下|晚)午?)?\s*(\d{1,2})[点时:]\d{0,2}?[到至—\-](\d{1,2})[点时:]?",
            text,
        )
        if range_match:
            h1 = int(range_match.group(2))
            h2 = int(range_match.group(3))
            if "下" in text or "晚" in text:
                h1 = h1 + 12 if h1 < 12 else h1
                h2 = h2 + 12 if h2 < 12 else h2
            start_time = f"{h1:02d}:00"
            end_time = f"{h2:02d}:00"
        elif start_time:
            # 默认 2 小时
            hour = int(start_time.split(":")[0])
            end_hour = (hour + 2) % 24
            end_time = f"{end_hour:02d}:00"

    return {
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "area": area,
    }
