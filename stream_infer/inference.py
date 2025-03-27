import threading as th
import gc
import time
from typing import Union, List, Literal, Dict, Tuple, Optional, Callable, Any
from functools import lru_cache

from .dispatcher import Dispatcher
from .algo import BaseAlgo
from .player import Player
from .recorder import Recorder
from .timer import Timer
from .model import Mode
from .log import logger


class Inference:
    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher
        self.inferences_info: List[Tuple[BaseAlgo, int, int, Union[int, float]]] = []
        self.timers: Dict[str, Timer] = {}

        self.is_stop = False
        self.process_func: Callable = self.default_process
        self._last_gc_time = time.time()
        self._gc_interval = 60  # 每60秒执行一次垃圾回收

    def load_algo(
        self,
        algo_instance: BaseAlgo,
        frame_count: int,
        frame_step: int,
        interval: Union[int, float],
        **kwargs,
    ) -> None:
        """加载算法实例

        Args:
            algo_instance: 算法实例，必须是BaseAlgo的子类
            frame_count: 处理的帧数
            frame_step: 帧步长
            interval: 执行间隔（秒）
            kwargs: 其他参数

        Raises:
            ValueError: 如果algo_instance不是BaseAlgo的实例
        """
        if not isinstance(algo_instance, BaseAlgo):
            err = f"Algo instance must be an instance of `BaseAlgo`, but got {type(algo_instance)}"
            logger.error(err)
            raise ValueError(err)

        try:
            self.inferences_info.append(
                (algo_instance, frame_count, frame_step, interval)
            )
            self.timers[algo_instance.name] = Timer(interval, key=algo_instance.name)
            algo_instance.init(**kwargs)
            logger.info(f"Successfully loaded algorithm: {algo_instance.name}")
        except Exception as e:
            logger.error(f"Failed to load algorithm {algo_instance.name}: {str(e)}")
            raise

    def list_algos(self) -> List[str]:
        """列出所有已加载的算法名称

        Returns:
            List[str]: 算法名称列表
        """
        return [algo_instance.name for algo_instance, _, _, _ in self.inferences_info]

    def run(self) -> None:
        """运行所有满足时间条件的算法"""
        current_time = time.time()

        # 定期执行垃圾回收
        if current_time - self._last_gc_time > self._gc_interval:
            gc.collect()
            self._last_gc_time = current_time
            logger.debug("Performed garbage collection")

        for inference_info in self.inferences_info:
            algo_instance, _, _, _ = inference_info
            timer = self.timers[algo_instance.name]
            if timer.is_time():
                try:
                    self._infer(inference_info)
                except Exception as e:
                    logger.error(
                        f"Error running inference for {algo_instance.name}: {str(e)}"
                    )
                    # 继续执行其他算法，不让一个算法的失败影响整体

    def run_loop(self) -> None:
        """循环运行算法直到停止信号"""
        try:
            while not self.is_stop:
                self.run()
                # 添加短暂休眠以减少CPU使用率
                time.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in run loop: {str(e)}")
            self.is_stop = True

    def run_async(self) -> th.Thread:
        """异步运行算法

        Returns:
            Thread: 运行算法的线程
        """
        thread = th.Thread(
            target=self.run_loop, daemon=True
        )  # 使用daemon=True确保主程序退出时线程也会退出
        thread.start()
        return thread

    def stop(self) -> None:
        """停止算法运行"""
        self.is_stop = True
        logger.info("Stopping inference")

    def auto_run_specific(self, fps: int, current_frame_index: int) -> List[str]:
        """根据帧索引自动运行特定算法

        Args:
            fps: 每秒帧数
            current_frame_index: 当前帧索引

        Returns:
            List[str]: 运行的算法名称列表
        """
        current_algo_names = []
        for algo_instance, _, _, frequency in self.inferences_info:
            # 计算应该在哪些帧运行算法
            interval_frames = max(1, int(frequency * fps))
            if current_frame_index % interval_frames == 0:
                try:
                    self.run_specific(algo_instance.name)
                    current_algo_names.append(algo_instance.name)
                except Exception as e:
                    logger.error(
                        f"Error running algorithm {algo_instance.name}: {str(e)}"
                    )
        return current_algo_names

    def run_specific(self, algo_name: str) -> bool:
        """运行特定名称的算法

        Args:
            algo_name: 算法名称

        Returns:
            bool: 是否成功运行算法
        """
        for inference_info in self.inferences_info:
            algo_instance, _, _, _ = inference_info
            if algo_instance.name == algo_name:
                try:
                    self._infer(inference_info)
                    return True
                except Exception as e:
                    logger.error(f"Error running algorithm {algo_name}: {str(e)}")
                    return False
        logger.warning(f"Algorithm {algo_name} not found")
        return False

    # 使用较小的缓存大小，避免占用过多内存
    @lru_cache(maxsize=8)  # 缓存最近的推理结果，避免重复计算
    def _get_cached_result(self, algo_name: str, frame_hash: int) -> Any:
        """获取缓存的推理结果（内部方法）

        Args:
            algo_name: 算法名称
            frame_hash: 帧的哈希值，用于标识帧内容

        Returns:
            Any: 缓存的结果，如果没有缓存则返回None
        """
        # 这个方法本身不做任何事情，只是作为lru_cache的包装
        # 实际上永远不会被调用，因为lru_cache会拦截调用并返回缓存的结果
        return None

    def _infer(self, inference_info) -> int:
        """执行推理（内部方法）

        Args:
            inference_info: 推理信息元组 (algo_instance, frame_count, frame_step, interval)

        Returns:
            int: 状态码，-1表示失败，0表示成功
        """
        algo_instance, frame_count, frame_step, _ = inference_info
        try:
            # 获取帧
            frames = self.dispatcher.get_frames(frame_count, frame_step)
            if not frames:
                logger.warning(f"No frames available for {algo_instance.name}")
                return -1

            # 计算帧内容的哈希值，用于缓存查找
            # 使用帧的id作为哈希值，避免创建大量临时字符串对象
            frame_hash = id(frames[0])

            # 尝试从缓存获取结果
            cached_result = self._get_cached_result(algo_instance.name, frame_hash)
            if cached_result is not None:
                logger.debug(f"Using cached result for {algo_instance.name}")
                result = cached_result
            else:
                # 执行算法
                start_time = time.time()
                result = algo_instance.run(frames)
                elapsed = time.time() - start_time
                logger.debug(
                    f"Algorithm {algo_instance.name} took {elapsed:.4f}s to run"
                )

                # 缓存结果，让LRU机制自动管理
                self._get_cached_result(algo_instance.name, frame_hash)

            # 收集结果
            self.dispatcher.collect(
                self.dispatcher.get_current_position(), algo_instance.name, result
            )
            return 0
        except Exception as e:
            logger.error(f"Error in inference for {algo_instance.name}: {str(e)}")
            return -1

    def default_process(self, *args, **kwargs) -> None:
        """默认的处理函数，不做任何事情"""
        pass

    def process(self, func: Callable) -> Callable:
        """设置处理函数

        Args:
            func: 处理函数

        Returns:
            Callable: 包装后的处理函数
        """

        def wrapper(*args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in process function: {str(e)}")
                return None

        self.process_func = wrapper
        return wrapper

    def start(
        self,
        player: Player,
        fps: int = 30,
        position: int = 0,
        mode: Literal["realtime", "offline"] = "realtime",
        recording_path: Optional[str] = None,
        logging_level: str = "INFO",
    ) -> None:
        """启动推理

        Args:
            player: 播放器实例
            fps: 每秒帧数
            position: 起始位置（秒）
            mode: 模式，'realtime'或'offline'
            recording_path: 录制路径，如果为None则不录制
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'

        Raises:
            ValueError: 如果mode不是支持的模式
        """
        # 设置日志级别
        from stream_infer.log import set_log_level

        # 在主进程中设置日志级别
        set_log_level(logging_level)

        try:
            if mode in [Mode.OFFLINE, Mode.OFFLINE.value]:
                self._start_offline_mode(player, fps, position, recording_path)
            elif mode in [Mode.REALTIME, Mode.REALTIME.value]:
                # 传递日志级别参数
                self._start_realtime_mode(player, fps, logging_level=logging_level)
            else:
                err = f"Unsupported mode: {mode}, only support `realtime` or `offline`"
                logger.error(err)
                raise ValueError(err)
        finally:
            # 确保资源被清理
            self.dispatcher.clear()
            gc.collect()

    def _start_offline_mode(
        self, player: Player, fps: int, position: int, recording_path: Optional[str]
    ) -> None:
        """启动离线模式

        Args:
            player: 播放器实例
            fps: 每秒帧数
            position: 起始位置（秒）
            recording_path: 录制路径，如果为None则不录制
        """
        recorder = None
        try:
            if recording_path:
                recorder = Recorder(player, recording_path)

            # 创建一个弱引用字典来跟踪已处理的帧
            import weakref

            processed_frames = weakref.WeakValueDictionary()

            for frame, current_frame_index in player.play(fps, position):
                # 运行算法并处理帧
                current_algo_names = self.auto_run_specific(fps, current_frame_index)
                try:
                    # 创建帧的副本进行处理，避免修改原始帧
                    import numpy as np

                    frame_copy = np.copy(frame) if frame is not None else None

                    processed_frame = self.process_func(
                        frame=frame_copy, current_algo_names=current_algo_names
                    )
                    frame_to_use = (
                        processed_frame if processed_frame is not None else frame_copy
                    )
                except Exception as e:
                    logger.error(f"Error in process function: {str(e)}")
                    frame_to_use = frame

                # 录制处理后的帧
                if recorder:
                    recorder.add_frame(frame_to_use)

                # 将处理过的帧添加到弱引用字典中
                processed_frames[current_frame_index] = frame_to_use

                # 定期执行垃圾回收
                if current_frame_index % (fps * 5) == 0:  # 每5秒执行一次垃圾回收
                    # 清除缓存以释放内存
                    self._get_cached_result.cache_clear()
                    # 强制垃圾回收
                    gc.collect()
                    # 记录内存使用情况
                    logger.debug(
                        f"Performed garbage collection in offline mode, frame index: {current_frame_index}, processed frames: {len(processed_frames)}"
                    )

                # 主动释放不再需要的帧引用
                frame = None
                frame_copy = None
                frame_to_use = None
        finally:
            # 确保录制器被关闭
            if recorder:
                recorder.close()

    def _start_realtime_mode(
        self, player: Player, fps: int, logging_level: str = "INFO"
    ) -> None:
        """启动实时模式

        Args:
            player: 播放器实例
            fps: 每秒帧数
            logging_level: 日志级别，可选值为'DEBUG', 'INFO', 'WARNING', 'ERROR'，默认为'INFO'
        """
        try:
            # 启动异步播放和推理
            player_thread = player.play_async(fps, logging_level=logging_level)
            inference_thread = self.run_async()

            # 创建一个弱引用字典来跟踪已处理的帧
            import weakref

            processed_frames = weakref.WeakValueDictionary()

            # 监控播放器状态
            last_gc_time = time.time()
            frame_count = 0
            while player.is_active():
                try:
                    # 获取当前帧（如果有）
                    current_frame = None
                    try:
                        # 在实时模式下，dispatcher是一个代理对象，不能直接访问_lock
                        frames = self.dispatcher.get_frames(1, 0)  # 获取最新的一帧
                        if frames and len(frames) > 0:
                            # 创建帧的副本进行处理，避免修改原始帧
                            import numpy as np

                            current_frame = (
                                np.copy(frames[0]) if frames[0] is not None else None
                            )
                    except Exception as e:
                        logger.error(f"Error getting frames: {str(e)}")

                    # 处理帧
                    if current_frame is not None:
                        processed_frame = self.process_func(frame=current_frame)
                        if processed_frame is not None:
                            # 将处理过的帧添加到弱引用字典中
                            processed_frames[frame_count] = processed_frame

                        # 释放引用
                        current_frame = None
                        processed_frame = None

                    # 短暂休眠以减少CPU使用率
                    time.sleep(0.001)

                    # 定期执行垃圾回收
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_gc_time > 5:  # 每5秒执行一次垃圾回收
                        # 清除缓存以释放内存
                        self._get_cached_result.cache_clear()
                        # 强制垃圾回收
                        gc.collect()
                        last_gc_time = current_time
                        # 记录内存使用情况
                        logger.debug(
                            f"Performed garbage collection in realtime mode, frame count: {frame_count}, processed frames: {len(processed_frames)}"
                        )
                except Exception as e:
                    logger.error(f"Error in process function: {str(e)}")
        finally:
            # 确保资源被清理
            self.stop()
            player.stop()
