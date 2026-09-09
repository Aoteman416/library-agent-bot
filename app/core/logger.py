"""日志配置"""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "app") -> logging.Logger:
    """获取统一配置的 logger（单次初始化）"""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        _CONFIGURED = True
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("app")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
    return logger


logger = get_logger("app")
