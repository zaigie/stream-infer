import time
import multiprocessing as mp
from multiprocessing.managers import BaseProxy

from .timer import Timer
from .log import logger


class Player:
    def __init__(self, dispatcher, producer, path):
        self.dispatcher = dispatcher
        self.producer = producer
        self.path = path
        try:
            self.info = self.producer.get_info(self.path)
        except Exception as e:
            raise ValueError(f"Error getting info: {e}")
        self.fps = self.info["fps"]
        self.play_fps = self.fps
        self.frame_count = self.info["frame_count"]
        self.process = None
        self.is_end = mp.Value("b", False)

    def play(self, fps=None):
        fps = self.fps if fps is None else fps
        self.play_fps = fps
        interval_count = 0

        for idx, frame in enumerate(self.producer.read(self.path, fps)):
            self.dispatcher.add_frame(frame)
            interval_count += 1
            if interval_count >= fps:
                interval_count = 0
                self.dispatcher.increase_current_time()
                logger.debug(f"current time: {self.get_play_time()}")
            yield frame, self.dispatcher.get_current_frame()

    def play_async(self, fps=None):
        """
        Starts the appropriate streaming process based on the frame count.
        """
        if not isinstance(self.dispatcher, BaseProxy):
            logger.error(
                f"Dispatcher is not an proxy: {type(self.dispatcher)}, use DispatcherManager().create() to create one"
            )
            raise ValueError(
                f"Dispatcher is not an proxy: {type(self.dispatcher)}, use DispatcherManager().create() to create one"
            )

        self.rt_frames = mp.Queue()
        if fps is None or fps >= self.fps:
            fps = self.fps
            if fps > 30:
                logger.warning(
                    f"FPS {fps} is too high, if your player is playing more slowly than the actual time, set a lower play fps"
                )
        self.play_fps = fps

        if self.frame_count == -1:
            target = self.normal_stream
        else:
            target = self.video_stream

        self.process = mp.Process(target=target)
        self.process.start()
        return self.process

    def stop(self):
        if self.process:
            self.is_end.value = True
        self.process.terminate()

    def is_active(self) -> bool:
        """
        Checks if the streaming process is still running.
        """
        return (
            self.process.is_alive() and not self.is_end.value if self.process else False
        )

    def get_play_time(self) -> str:
        current_time = self.dispatcher.get_current_time()
        return f"{current_time // 3600:02d}:{current_time // 60 % 60:02d}:{current_time % 60:02d}"

    def video_stream(self):
        """
        Handles streaming for video files. Frames are processed at a rate determined by the video's FPS.
        """
        base_interval = 1 / self.play_fps
        start_time = time.time()
        interval_count = 0

        for idx, frame in enumerate(self.producer.read(self.path, self.play_fps)):
            target_time = start_time + (idx * base_interval)
            time.sleep(max(0, target_time - time.time()))
            self.dispatcher.add_frame(frame)
            self.rt_frames.put(frame)
            interval_count += 1
            if interval_count >= self.play_fps:
                interval_count = 0
                self.dispatcher.increase_current_time()
                logger.debug(f"current time: {self.get_play_time()}")

        self.is_end.value = True

    def normal_stream(self):
        """
        Handles streaming for non-video files. Frames are processed at regular intervals.
        """
        timer = Timer(interval=1)
        for frame in self.producer.read(self.path, self.play_fps):
            if timer.is_time():
                self.dispatcher.increase_current_time()
                logger.debug(f"current time: {self.get_play_time()}")
            self.dispatcher.add_frame(frame)
            self.rt_frames.put(frame)

        self.is_end.value = True
