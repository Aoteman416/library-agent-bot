"""日志配置"""
import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str = "app") -> logging.Logger:
    """获取统一配置的 logger（单次初始化）

    日志级别可用环境变量 LOG_LEVEL 覆盖（评测脚本会把安全拦截日志压到 WARNING 以上，
    避免几千条拦截记录淹没指标输出）。
    """
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        _CONFIGURED = True
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("app")
        root.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
        root.addHandler(handler)
    return logger


logger = get_logger("app")
