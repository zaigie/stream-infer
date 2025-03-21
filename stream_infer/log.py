import sys
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


def set_log_level(level="INFO"):
    """
    设置 stream_infer 库的日志级别
    
    Args:
        level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
    """
    global logger, _internal_logger
    
    # 更新当前进程的日志级别
    _internal_logger.remove()
    _internal_logger.add(sys.stdout, format=_LOG_FORMAT, level=level.upper())
