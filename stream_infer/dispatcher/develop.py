from typing import Dict, Any, Optional, Tuple
import multiprocessing as mp
from collections import defaultdict

from .base import Dispatcher, Mode
from ..log import logger


class DevelopDispatcher(Dispatcher):
    """
    开发环境使用的调度器，扩展了基础调度器功能，增加了结果存储和查询能力

    Attributes:
        collect_results: 存储算法结果的字典，格式为 {algo_name: {position: result}}
        _results_lock: 保护结果字典的线程锁
    """

    def __init__(self, mode: Mode, buffer: int, logging_level: str = "INFO", **kwargs):
        """
        初始化开发调度器

        Args:
            mode: 模式
            buffer: 缓冲区大小
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
            **kwargs: 其他参数
        """
        super().__init__(mode, buffer, logging_level, **kwargs)
        self.collect_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._results_lock = mp.Lock()  # 使用多进程锁替代线程锁

    def collect(self, position: int, algo_name: str, result: Any) -> None:
        """
        收集算法结果并存储

        Args:
            position: 位置（秒）
            algo_name: 算法名称
            result: 算法结果
        """
        # 调用父类方法记录日志
        super().collect(position, algo_name, result)

        # 线程安全地存储结果
        with self._results_lock:
            self.collect_results[algo_name][str(position)] = result

    def get_result(self, name: str, clear: bool = False) -> Optional[Dict[str, Any]]:
        """
        获取指定算法的所有结果

        Args:
            name: 算法名称
            clear: 是否在获取后清除结果

        Returns:
            Optional[Dict[str, Any]]: 算法结果字典，如果不存在则返回None
        """
        with self._results_lock:
            if name in self.collect_results and self.collect_results[name]:
                # 创建深拷贝以避免并发修改问题
                data = dict(self.collect_results[name])
                if clear:
                    del self.collect_results[name]
                return data
        return None

    def get_last_result(
        self, name: str, clear: bool = False
    ) -> Tuple[int, Optional[Any]]:
        """
        获取指定算法的最新结果

        Args:
            name: 算法名称
            clear: 是否在获取后清除结果

        Returns:
            Tuple[int, Optional[Any]]: (时间点, 结果)，如果不存在则返回(-1, None)
        """
        algo_results = self.get_result(name, clear)
        if algo_results and algo_results.keys():
            try:
                # 找出最大的时间点（最新的结果）
                time_point = max(int(k) for k in algo_results.keys())
                return time_point, algo_results[str(time_point)]
            except (ValueError, KeyError) as e:
                logger.error(f"Error getting last result for {name}: {str(e)}")
        return -1, None

    def clear(self) -> None:
        """
        清空调度器状态和结果
        """
        super().clear()
        with self._results_lock:
            self.collect_results.clear()
