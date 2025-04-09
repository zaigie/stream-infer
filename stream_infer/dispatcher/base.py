from multiprocessing.managers import BaseManager
from collections import deque
import threading
from typing import List, Any, Literal, Optional, Deque, Union, TypeVar

from ..util import position2time
from ..log import logger
from ..model import Mode

# 定义泛型类型
T = TypeVar("T")


class Dispatcher:
    """
    调度器基类，负责管理帧缓冲区和收集算法结果

    Attributes:
        queue: 帧缓冲队列
        current_position: 当前位置（秒）
        current_frame_index: 当前帧索引
        _lock: 线程锁，用于保护队列操作的线程安全
    """

    def __init__(self, mode: Mode, buffer: int, logging_level: str = "INFO", **kwargs):
        """
        初始化调度器

        Args:
            mode: 模式，'realtime'或'offline'
            buffer: 缓冲区大小（帧数）
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
            **kwargs: 其他参数
        """
        # 在子进程中设置日志级别
        from stream_infer.log import set_log_level

        set_log_level(logging_level)

        self.buffer_size: int = buffer
        self.queue: Deque[Any] = deque(maxlen=buffer)
        self.current_position: int = 0
        self.current_frame_index: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._mode: Optional[Mode] = mode
        self._log_level: str = logging_level

    def get_mode(self) -> Optional[Mode]:
        return self._mode

    def get_buffer_size(self) -> int:
        return self.buffer_size

    def get_log_level(self) -> str:
        return self._log_level

    def add_frame(self, frame: Any) -> None:
        """
        添加帧到缓冲区

        Args:
            frame: 要添加的帧
        """
        with self._lock:
            self.queue.append(frame)
            self.current_frame_index += 1

    def get_frames(self, count: int, step: int) -> List[Any]:
        """
        从缓冲区获取指定数量和步长的帧

        Args:
            count: 要获取的帧数
            step: 帧步长, 必须大于等于0

        Returns:
            List[Any]: 帧列表

        Raises:
            ValueError: 如果step小于0
        """
        if step < 0:
            raise ValueError("step must be positive or zero")

        with self._lock:
            queue_len = len(self.queue)

            # 如果队列为空或者帧不足，返回空列表
            if queue_len == 0 or (step > 0 and queue_len < count * step):
                return []

            if step == 0:
                # 如果step为0，直接返回最近的count帧
                return list(self.queue)[-min(count, queue_len) :]
            else:
                # 优化：避免多次列表转换和切片操作
                result = []
                # 从最新的帧开始，按照step步长获取count个帧
                indices = range(
                    queue_len - 1, max(-1, queue_len - count * step - 1), -step
                )
                for i in indices[:count]:
                    if i >= 0:
                        result.append(self.queue[i])
                return result[::-1]

    def collect(self, position: int, algo_name: str, result: Any) -> None:
        """
        收集算法结果

        Args:
            position: 位置（秒）
            algo_name: 算法名称
            result: 算法结果
        """
        logger.debug(
            f"[{position2time(position)}] collect {algo_name} result: {result}"
        )

    def increase_current_position(self) -> None:
        """
        增加当前位置（秒）
        """
        with self._lock:
            self.current_position += 1

    def get_current_position(self) -> int:
        """
        获取当前位置（秒）

        Returns:
            int: 当前位置
        """
        return self.current_position

    def set_current_position(self, time_pos: int) -> None:
        """
        设置当前位置（秒）

        Args:
            time_pos: 要设置的位置
        """
        with self._lock:
            self.current_position = time_pos

    def get_current_frame_index(self) -> int:
        """
        获取当前帧索引

        Returns:
            int: 当前帧索引
        """
        return self.current_frame_index

    def set_current_frame_index(self, frame_idx: int) -> None:
        """
        设置当前帧索引

        Args:
            frame_idx: 要设置的帧索引
        """
        with self._lock:
            self.current_frame_index = frame_idx

    def clear(self) -> None:
        """
        清空调度器状态
        """
        with self._lock:
            self.queue.clear()
            self.current_position = 0
            self.current_frame_index = 0

    @classmethod
    def create(
        cls,
        mode: Literal["realtime", "offline"] = "realtime",
        buffer: int = 30,
        logging_level: str = "INFO",
        **kwargs,
    ) -> Union["Dispatcher", Any]:
        """
        创建调度器实例

        Args:
            mode: 模式，'realtime'或'offline'
            buffer: 缓冲区大小
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
            **kwargs: 其他参数

        Returns:
            Union[Dispatcher, Any]: 调度器实例

        Raises:
            ValueError: 如果mode不是支持的模式
        """
        try:
            if mode in [Mode.OFFLINE, Mode.OFFLINE.value]:
                return cls(Mode.OFFLINE, buffer, logging_level, **kwargs)
            elif mode in [Mode.REALTIME, Mode.REALTIME.value]:
                return DispatcherManager(cls).create(
                    Mode.REALTIME, buffer, logging_level, **kwargs
                )
            else:
                err = f"Unsupported mode: {mode}, only support `realtime` or `offline`"
                logger.error(err)
                raise ValueError(err)
        except Exception as e:
            logger.error(f"Error creating dispatcher: {str(e)}")
            raise


class DispatcherManager:
    """
    调度器管理器，用于创建多进程共享的调度器实例

    Attributes:
        _manager: 进程管理器
        _dispatcher: 调度器实例
        _obj: 调度器类
    """

    def __init__(self, obj: Optional[type] = None):
        """
        初始化调度器管理器

        Args:
            obj: 调度器类, 如果为None则使用Dispatcher
        """
        self._manager: Optional[BaseManager] = None
        self._dispatcher: Optional[Any] = None
        self._obj: type = Dispatcher if obj is None else obj

    def create(
        self, mode: Mode, buffer: int, logging_level: str = "INFO", **kwargs
    ) -> Any:
        """
        创建调度器实例

        Args:
            mode: 模式
            buffer: 缓冲区大小
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
            **kwargs: 其他参数

        Returns:
            Any: 调度器实例
        """
        try:
            if self._manager is None:
                self._initialize_manager(mode, buffer, logging_level, **kwargs)
            return self._dispatcher
        except Exception as e:
            logger.error(f"Error creating dispatcher manager: {str(e)}")
            raise

    def _initialize_manager(
        self, mode: Mode, buffer: int, logging_level: str = "INFO", **kwargs
    ) -> None:
        """
        初始化进程管理器

        Args:
            mode: 模式
            buffer: 缓冲区大小
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
            **kwargs: 其他参数
        """
        try:
            BaseManager.register("Dispatcher", self._obj)
            self._manager = BaseManager()
            self._manager.start()
            self._dispatcher = self._manager.Dispatcher(
                mode, buffer, logging_level, **kwargs
            )
            logger.debug("Successfully initialized dispatcher manager")
        except Exception as e:
            logger.error(f"Error initializing dispatcher manager: {str(e)}")
            raise
