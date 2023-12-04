import time
import multiprocessing as mp
from multiprocessing.managers import BaseProxy

from .timer import Timer
from .log import logger


class Player:
    def __init__(self, producer, frame_tracker, path):
        self.frame_tracker = frame_tracker
        self.producer = producer
        self.path = path
        self.process = None
        self.fps = None
        self.current_frame = 0

    def play(self, fps=None):
        try:
            original_fps = self.producer.get_info(self.path)["fps"]
            if fps is None:
                fps = original_fps
            self.fps = fps
        except Exception as e:
            raise ValueError(f"Error getting fps: {e}")

        interval_count = 0

        for frame in self.producer.read(self.path, fps):
            self.frame_tracker.add_frame(frame)
            interval_count += 1
            if interval_count >= fps:
                interval_count = 0
                self.frame_tracker.increase_current_time()
                logger.debug(f"current time: {self.get_current_time_str()}")
            self.current_frame += 1
            yield frame, self.current_frame

    def play_realtime(self, fps=None):
        """
        Starts the appropriate streaming process based on the frame count.
        """
        if not isinstance(self.frame_tracker, BaseProxy):
            logger.error(
                f"Frame tracker is not an proxy: {type(self.frame_tracker)}, use TrackerManager().create() to create one"
            )
            raise ValueError(
                f"Frame tracker is not an proxy: {type(self.frame_tracker)}, use TrackerManager().create() to create one"
            )
        try:
            info = self.producer.get_info(self.path)
            frame_count = info["frame_count"]
            original_fps = info["fps"]
            if fps is None or fps >= original_fps:
                fps = original_fps
                if fps > 30:
                    logger.warning(
                        f"FPS {fps} is too high, if your player is playing more slowly than the actual time, set a lower fps"
                    )
            self.fps = fps
        except Exception as e:
            raise ValueError(f"Error getting info: {e}")

        if frame_count == -1:
            target = self.normal_stream
        else:
            target = self.video_stream

        self.process = mp.Process(
            target=target, args=(self.frame_tracker, self.producer, self.path)
        )
        self.process.start()

    def is_active(self) -> bool:
        """
        Checks if the streaming process is still running.
        """
        return self.process.is_alive() if self.process else False

    def get_current_time_str(self) -> str:
        current_time = self.frame_tracker.get_current_time()
        return f"{current_time // 3600:02d}:{current_time // 60 % 60:02d}:{current_time % 60:02d}"

    def video_stream(self, frame_tracker, producer, path):
        """
        Handles streaming for video files. Frames are processed at a rate determined by the video's FPS.
        """
        base_interval = 1 / self.fps
        start_time = time.time()
        interval_count = 0

        for idx, frame in enumerate(producer.read(path, self.fps)):
            target_time = start_time + (idx * base_interval)
            time.sleep(max(0, target_time - time.time()))

            frame_tracker.add_frame(frame)
            interval_count += 1
            if interval_count >= self.fps:
                interval_count = 0
                frame_tracker.increase_current_time()
                logger.debug(f"current time: {self.get_current_time_str()}")

    def normal_stream(self, frame_tracker, producer, path):
        """
        Handles streaming for non-video files. Frames are processed at regular intervals.
        """
        timer = Timer(interval=1)
        for frame in producer.read(path, self.fps):
            if timer.is_time():
                frame_tracker.increase_current_time()
                logger.debug(f"current time: {self.get_current_time_str()}")
            frame_tracker.add_frame(frame)
