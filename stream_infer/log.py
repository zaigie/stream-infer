import sys
import threading
from loguru import logger as _logger

# 默认日志格式
_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# 创建一个库内部独立的logger
_internal_logger = _logger.bind(library="stream_infer")

# 初始化日志配置，默认级别为INFO
_internal_logger.remove()
_internal_logger.add(sys.stdout, format=_LOG_FORMAT, level="INFO")

# 导出库内部使用的logger
logger = _internal_logger

# 添加线程锁以确保线程安全
_log_lock = threading.Lock()


def set_log_level(level="INFO"):
    """
    设置 stream_infer 库的日志级别

    Args:
        level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
    """
    global logger, _internal_logger

    # 标准化日志级别
    level = level.upper()
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        level = "INFO"

    # 使用线程锁确保线程安全
    with _log_lock:
        # 更新当前进程的日志级别
        _internal_logger.remove()
        _internal_logger.add(sys.stdout, format=_LOG_FORMAT, level=level)

        # 记录日志级别变更
        # _internal_logger.debug(f"Log level set to {level}")
