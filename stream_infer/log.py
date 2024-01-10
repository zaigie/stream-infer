import os
import sys
from loguru import logger


def get_logger():
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stdout, format=log_format, level="DEBUG")
    return logger
