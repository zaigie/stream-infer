import threading as th
import gc
import time
import os
import sys
import traceback
import multiprocessing as mp
from typing import Union, List, Dict, Tuple, Optional, Callable, Any, Dict
from functools import lru_cache

from .dispatcher import Dispatcher
from .algo import BaseAlgo
from .player import Player
from .recorder import Recorder
from .timer import Timer
from .model import Mode
from .log import logger


def _base_algo_process(
    algo_instance,
    dispatcher,
    frame_count,
    frame_step,
    interval,
    stop_event,
    mode="realtime",  # "realtime" 或 "offline"
    fps=None,
):
    """算法进程的基础函数，支持实时和离线模式

    Args:
        algo_instance: 算法实例
        dispatcher: 调度器实例
        frame_count: 处理的帧数
        frame_step: 帧步长
        interval: 执行间隔（秒）
        stop_event: 停止事件
        mode: 运行模式，"realtime" 或 "offline"
        fps: 每秒帧数（仅离线模式需要）
    """
    # 确保子进程中的打印能够立即显示
    import sys

    # 禁用输出缓冲，确保打印立即显示
    sys.stdout.flush()
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    # 在子进程中设置日志级别
    from stream_infer.log import set_log_level

    set_log_level(os.environ.get("STREAM_INFER_LOG_LEVEL", "INFO"))

    # 在子进程中初始化算法
    # 检查算法实例是否有初始化参数
    init_kwargs = getattr(algo_instance, "_init_kwargs", {})
    try:
        logger.info(f"Initializing algorithm {algo_instance.name} in {mode} process")
        algo_instance.init(**init_kwargs)
        logger.info(f"Algorithm {algo_instance.name} initialized in {mode} process")
    except Exception as e:
        logger.error(
            f"Failed to initialize algorithm {algo_instance.name} in {mode} process: {str(e)}"
        )

    # 缓存最近的推理结果
    from functools import lru_cache

    @lru_cache(maxsize=8)
    def _get_cached_result(algo_name, frame_hash):
        return None

    last_gc_time = time.time()
    gc_interval = 60  # 每60秒执行一次垃圾回收
    last_frame_index = -1

    # 离线模式需要计算帧间隔
    if mode == "offline" and fps is not None:
        interval_frames = max(1, int(interval * fps))

    # 实时模式使用定时器
    timer = None
    if mode == "realtime":
        timer = Timer(interval, key=algo_instance.name)

    try:
        while not stop_event.is_set():
            try:
                # 离线模式下的帧选择逻辑
                if mode == "offline":
                    # 获取当前帧索引
                    current_frame_index = dispatcher.get_current_frame_index()

                    # 如果没有新帧或者不是该算法应该运行的帧，则跳过
                    if (
                        current_frame_index <= last_frame_index
                        or current_frame_index % interval_frames != 0
                    ):
                        time.sleep(0.0001)
                        continue

                    last_frame_index = current_frame_index
                    # 添加调试日志，帮助追踪进程执行
                    logger.debug(f"处理帧索引 {current_frame_index} 在 {mode} 模式下，算法: {algo_instance.name}")
                # 实时模式下的定时器逻辑
                elif mode == "realtime" and not timer.is_time():
                    # 使用更短的睡眠时间，提高响应速度
                    time.sleep(0.0001)
                    continue

                # 获取帧
                frames = dispatcher.get_frames(frame_count, frame_step)
                if not frames:
                    time.sleep(0.0001)
                    continue
                
                # 添加调试日志，确认帧获取成功
                logger.debug(f"成功获取 {len(frames)} 帧用于算法 {algo_instance.name}")

                # 计算帧内容的哈希值
                frame_hash = id(frames[0])

                # 尝试从缓存获取结果
                cached_result = _get_cached_result(algo_instance.name, frame_hash)
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

                    # 缓存结果
                    _get_cached_result(algo_instance.name, frame_hash)

                # 收集结果
                dispatcher.set_last_algo_name(algo_instance.name)
                dispatcher.collect(
                    dispatcher.get_current_position(), algo_instance.name, result
                )
                
                # 添加调试日志，确认结果已收集
                logger.debug(f"算法 {algo_instance.name} 已收集结果，当前帧位置: {dispatcher.get_current_position()}")
                # 立即刷新标准输出缓冲区，确保输出可见
                sys.stdout.flush()

                # 离线模式下的结果已通过dispatcher.collect收集，无需额外处理

            except Exception as e:
                error_msg = f"Error in {'offline' if mode == 'offline' else ''} inference for {algo_instance.name}: {str(e)}"
                logger.error(error_msg)
                # 打印详细的错误堆栈信息
                traceback.print_exc()
                # 确保错误信息立即显示
                sys.stdout.flush()
                sys.stderr.flush()
                if mode == "offline":
                    time.sleep(0.01)  # 离线模式出错时稍微等待长一点，避免频繁报错

            # 使用更短的睡眠时间，提高响应速度
            time.sleep(0.0001)

            # 定期执行垃圾回收
            current_time = time.time()
            if current_time - last_gc_time > gc_interval:
                gc.collect()
                _get_cached_result.cache_clear()
                last_gc_time = current_time
                logger.debug(
                    f"Performed garbage collection in {mode} process for {algo_instance.name}"
                )
    except Exception as e:
        logger.error(
            f"Fatal error in {mode} algorithm process for {algo_instance.name}: {str(e)}"
        )
        # 打印详细的错误堆栈信息
        traceback.print_exc()
        # 确保错误信息立即显示
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        logger.info(
            f"{mode.capitalize()} algorithm process for {algo_instance.name} stopped"
        )
        # 确保最终信息立即显示
        sys.stdout.flush()
        sys.stderr.flush()


def _algo_process(
    algo_instance,
    dispatcher,
    frame_count,
    frame_step,
    interval,
    stop_event,
):
    """在单独进程中运行算法的函数（实时模式）

    Args:
        algo_instance: 算法实例
        dispatcher: 调度器实例
        frame_count: 处理的帧数
        frame_step: 帧步长
        interval: 执行间隔（秒）
        stop_event: 停止事件
    """
    return _base_algo_process(
        algo_instance,
        dispatcher,
        frame_count,
        frame_step,
        interval,
        stop_event,
        mode="realtime",
    )


def _offline_algo_process(
    algo_instance,
    dispatcher,
    frame_count,
    frame_step,
    interval,
    fps,
    stop_event,
):
    """在单独进程中运行算法的函数（离线模式）

    Args:
        algo_instance: 算法实例
        dispatcher: 调度器实例
        frame_count: 处理的帧数
        frame_step: 帧步长
        interval: 执行间隔（秒）
        fps: 每秒帧数
        stop_event: 停止事件
    """
    return _base_algo_process(
        algo_instance,
        dispatcher,
        frame_count,
        frame_step,
        interval,
        stop_event,
        frames_loaded_event,
        mode="offline",
        fps=fps,
    )


class Inference:
    def __init__(self, dispatcher: Dispatcher, player: Optional[Player] = None):
        self.dispatcher = dispatcher
        self.player = player
        self.inferences_info: List[Tuple[BaseAlgo, int, int, Union[int, float]]] = []
        self.timers: Dict[str, Timer] = {}

        self.is_stop = False
        self.process_func: Callable = self.default_process
        self._last_gc_time = time.time()
        self._gc_interval = 60  # 每60秒执行一次垃圾回收

        # 多进程相关属性
        self._algo_processes: Dict[str, mp.Process] = {}
        self._stop_events: Dict[str, mp.Event] = {}

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
            # 将算法参数保存下来，但不进行初始化
            # 初始化将在实际运行时进行，避免序列化问题
            self.inferences_info.append(
                (algo_instance, frame_count, frame_step, interval)
            )
            self.timers[algo_instance.name] = Timer(interval, key=algo_instance.name)

            # 将算法初始化参数保存下来，但不进行初始化
            # 将参数保存到算法实例中，以便在子进程中使用
            algo_instance._init_kwargs = kwargs

            logger.info(
                f"Successfully loaded algorithm: {algo_instance.name} (initialization deferred)"
            )
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

    def _cleanup_processes(self) -> None:
        """清理所有算法进程"""
        for algo_name, stop_event in self._stop_events.items():
            if stop_event is not None:
                stop_event.set()

        for algo_name, process in self._algo_processes.items():
            if process is not None and process.is_alive():
                logger.info(f"Terminating process for algorithm {algo_name}")
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()

        # 清空进程和事件字典
        self._algo_processes.clear()
        self._stop_events.clear()

    def _check_and_restart_processes(self, mode="realtime", fps=None) -> None:
        """检查算法进程是否存活，如果不存活则重启

        Args:
            mode: 运行模式，"realtime" 或 "offline"
            fps: 每秒帧数（仅离线模式需要）
        """
        for algo_name, process in list(self._algo_processes.items()):
            if not process.is_alive():
                logger.warning(
                    f"Process for algorithm {algo_name} has died, restarting..."
                )
                # 重启进程
                for inference_info in self.inferences_info:
                    if inference_info[0].name == algo_name:
                        algo_instance, frame_count, frame_step, interval = (
                            inference_info
                        )
                        stop_event = mp.Event()
                        self._stop_events[algo_name] = stop_event

                        if mode == "realtime":
                            # 使用 'spawn' 启动方法可以避免一些序列化问题
                            ctx = mp.get_context("spawn")
                            process = ctx.Process(
                                target=_algo_process,
                                args=(
                                    algo_instance,
                                    self.dispatcher,
                                    frame_count,
                                    frame_step,
                                    interval,
                                    stop_event,
                                ),
                                daemon=True,
                            )
                        else:  # offline
                            # 使用 'spawn' 启动方法可以避免一些序列化问题
                            ctx = mp.get_context("spawn")
                            process = ctx.Process(
                                target=_offline_algo_process,
                                args=(
                                    algo_instance,
                                    self.dispatcher,
                                    frame_count,
                                    frame_step,
                                    interval,
                                    fps,
                                    stop_event,
                                ),
                                daemon=True,
                            )

                        process.start()
                        self._algo_processes[algo_name] = process
                        logger.info(f"Restarted process for algorithm {algo_name}")
                        break

    def stop(self) -> None:
        """停止算法运行"""
        self.is_stop = True
        self._cleanup_processes()
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
        if self.dispatcher.get_buffer_size() < frame_count * (
            frame_step if frame_step > 0 else 1
        ):
            logger.error(
                f"Dispatcher `buffer` size is too small for {algo_instance.name}, buffer_size={self.dispatcher.get_buffer_size()} needed more than {frame_count * (frame_step if frame_step > 0 else 1)}"
            )
            return -1
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
            self.dispatcher.set_last_algo_name(algo_instance.name)
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

        if self.dispatcher.get_mode() == Mode.REALTIME:
            logger.warning(
                "Realtime mode process function not support `current_algo_names`"
            )
        self.process_func = wrapper
        return wrapper

    def start(
        self,
        fps: int = 30,
        position: int = 0,
        recording_path: Optional[str] = None,
        parallel: bool = False,
    ) -> None:
        """启动推理

        Args:
            fps: 每秒帧数
            position: 起始位置（秒）
            recording_path: 录制路径，如果为None则不录制
            parallel: 是否启用多进程并行算法推理，默认为False

        Raises:
            ValueError: 如果mode不是支持的模式或player未设置
        """
        if self.player is None:
            err = "Player is not set. Please set player when initializing Inference."
            logger.error(err)
            raise ValueError(err)

        # 设置日志级别
        from stream_infer.log import set_log_level

        # 使用从dispatcher获取的日志级别
        actual_logging_level = self.dispatcher.get_log_level()

        # 在主进程中设置日志级别
        set_log_level(actual_logging_level)
        os.environ["STREAM_INFER_LOG_LEVEL"] = actual_logging_level

        # 确保在启动新的进程前清理旧的进程
        self._cleanup_processes()

        mode = self.dispatcher.get_mode()

        try:
            if mode in [Mode.OFFLINE, Mode.OFFLINE.value]:
                self._start_offline_mode(
                    self.player, fps, position, recording_path, parallel
                )
            elif mode in [Mode.REALTIME, Mode.REALTIME.value]:
                # 实时模式下，如果有多个算法且parallel为True，则自动启用多进程
                if recording_path:
                    logger.warning("Realtime mode not support recording")
                # 在实时模式下，parallel参数默认为True，除非显式设置为False
                realtime_parallel = parallel if len(self.inferences_info) > 1 else False
                self._start_realtime_mode(self.player, fps, realtime_parallel)
            else:
                err = f"Unsupported mode: {mode}, only support `realtime` or `offline`"
                logger.error(err)
                raise ValueError(err)
        finally:
            # 确保资源被清理
            self.dispatcher.clear()
            gc.collect()

    def _start_offline_mode(
        self,
        player: Player,
        fps: int,
        position: int,
        recording_path: Optional[str],
        parallel: bool = False,
    ) -> None:
        """启动离线模式

        Args:
            player: 播放器实例
            fps: 每秒帧数
            position: 起始位置（秒）
            recording_path: 录制路径，如果为None则不录制
            parallel: 是否启用多进程并行算法推理，默认为False
        """

        recorder = None
        try:
            if recording_path:
                recorder = Recorder(player, recording_path)

            # 创建一个弱引用字典来跟踪已处理的帧
            import weakref

            processed_frames = weakref.WeakValueDictionary()

            # 如果启用并行模式且有多个算法，则为每个算法创建一个单独的进程
            if parallel and len(self.inferences_info) > 1:
                logger.info(
                    f"Starting {len(self.inferences_info)} algorithm processes in parallel mode for offline analysis"
                )

                # 为每个算法创建一个单独的进程
                for inference_info in self.inferences_info:
                    algo_instance, frame_count, frame_step, interval = inference_info
                    algo_name = algo_instance.name

                    # 创建停止事件
                    stop_event = mp.Event()
                    self._stop_events[algo_name] = stop_event

                    # 创建并启动进程
                    # 使用 'spawn' 启动方法可以避免一些序列化问题
                    ctx = mp.get_context("spawn")
                    process = ctx.Process(
                        target=_offline_algo_process,
                        args=(
                            algo_instance,
                            self.dispatcher,
                            frame_count,
                            frame_step,
                            interval,
                            fps,
                            stop_event,
                        ),
                        daemon=True,
                    )
                    process.start()
                    self._algo_processes[algo_name] = process
                    logger.info(f"Started process for algorithm {algo_name}")

                # 创建一个事件来通知所有子进程帧已经全部加载完成
                frames_loaded_event = mp.Event()
                
                # 主循环处理帧
                for frame, current_frame_index in player.play(fps, position):
                    # 将帧添加到调度器缓冲区
                    self.dispatcher.add_frame(frame)

                    # 检查所有算法进程是否仍在运行
                    self._check_and_restart_processes(mode="offline", fps=fps)

                    # 简单假设所有算法都已处理当前帧
                    # 注意：这里我们假设所有算法都已处理当前帧
                    # 实际上，用户可以通过重写 Dispatcher 的 collect 方法来实现自定义的结果存储和检索
                    current_algo_names = [info[0].name for info in self.inferences_info]

                    try:
                        # 创建帧的副本进行处理，避免修改原始帧
                        import numpy as np

                        frame_copy = np.copy(frame) if frame is not None else None

                        processed_frame = self.process_func(
                            frame=frame_copy,
                            current_algo_names=current_algo_names,
                            last_algo_name=self.dispatcher.get_last_algo_name(),
                        )
                        frame_to_use = (
                            processed_frame
                            if processed_frame is not None
                            else frame_copy
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
                        gc.collect()
                        logger.debug(
                            f"Performed garbage collection in offline parallel mode, frame index: {current_frame_index}, processed frames: {len(processed_frames)}"
                        )

                    # 主动释放不再需要的帧引用
                    frame = None
                    frame_copy = None
                    frame_to_use = None

                # 通知所有子进程帧已经全部加载完成
                logger.info("All frames loaded, waiting for algorithm processes to complete...")
                
                # 等待所有算法进程完成处理
                # 注意：这里我们不立即停止进程，而是等待它们完成处理
                for algo_name, process in self._algo_processes.items():
                    logger.info(f"Waiting for algorithm {algo_name} to complete...")
                    # 设置一个较长的超时时间，确保进程有足够的时间完成
                    process.join(timeout=60.0)
                    if process.is_alive():
                        logger.warning(f"Algorithm {algo_name} is still running after timeout, terminating...")
                        process.terminate()
                    else:
                        logger.info(f"Algorithm {algo_name} completed successfully")
                
                # 停止所有算法进程
                for algo_name, stop_event in self._stop_events.items():
                    stop_event.set()
            else:
                # 使用传统的单线程模式
                logger.info(
                    "Starting algorithms in single thread mode for offline analysis"
                )
                # 初始化算法
                for algo_instance, _, _, _ in self.inferences_info:
                    algo_instance.init(**algo_instance._init_kwargs)
                    logger.info(f"Successfully loaded algorithm: {algo_instance.name}")

                for frame, current_frame_index in player.play(fps, position):
                    # 运行算法并处理帧
                    current_algo_names = self.auto_run_specific(
                        fps, current_frame_index
                    )
                    try:
                        # 创建帧的副本进行处理，避免修改原始帧
                        import numpy as np

                        frame_copy = np.copy(frame) if frame is not None else None

                        processed_frame = self.process_func(
                            frame=frame_copy,
                            current_algo_names=current_algo_names,
                            last_algo_name=self.dispatcher.get_last_algo_name(),
                        )
                        frame_to_use = (
                            processed_frame
                            if processed_frame is not None
                            else frame_copy
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
        self, player: Player, fps: int, parallel: bool = False
    ) -> None:
        """启动实时模式

        Args:
            player: 播放器实例
            fps: 每秒帧数
            parallel: 是否启用多进程并行算法推理，默认为False
        """
        # 设置日志级别
        from stream_infer.log import set_log_level

        set_log_level(os.environ.get("STREAM_INFER_LOG_LEVEL", "INFO"))

        try:
            # 启动异步播放
            player_thread = player.play_async(fps)

            if parallel and len(self.inferences_info) > 1:
                logger.info(
                    f"Starting {len(self.inferences_info)} algorithm processes in parallel mode"
                )
                # 为每个算法创建一个单独的进程
                for inference_info in self.inferences_info:
                    algo_instance, frame_count, frame_step, interval = inference_info
                    algo_name = algo_instance.name

                    # 创建停止事件
                    stop_event = mp.Event()
                    self._stop_events[algo_name] = stop_event

                    # 创建并启动进程
                    process = mp.Process(
                        target=_algo_process,
                        args=(
                            algo_instance,
                            self.dispatcher,
                            frame_count,
                            frame_step,
                            interval,
                            stop_event,
                        ),
                        daemon=True,
                    )
                    process.start()
                    self._algo_processes[algo_name] = process
                    logger.info(f"Started process for algorithm {algo_name}")
            else:
                # 使用传统的单线程模式
                logger.info("Starting algorithms in single thread mode")
                inference_thread = self.run_async()

            # 创建一个弱引用字典来跟踪已处理的帧
            import weakref

            processed_frames = weakref.WeakValueDictionary()

            # 监控播放器状态
            last_gc_time = time.time()
            frame_count = 0
            while player.is_active():
                try:
                    # 检查多进程状态
                    if parallel and len(self.inferences_info) > 1:
                        self._check_and_restart_processes(mode="realtime")

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
                        processed_frame = self.process_func(
                            frame=current_frame,
                            current_algo_names=None,
                            last_algo_name=self.dispatcher.get_last_algo_name(),
                        )
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
