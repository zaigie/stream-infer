import time
import multiprocessing as mp
from multiprocessing.managers import BaseManager

from .timer import Timer


class Player:
    def __init__(self, frame_tracker, producer, path):
        self.frame_tracker = frame_tracker
        self.producer = producer
        self.path = path
        self.process = None
        self.current_frame = 0

    def play(self):
        try:
            fps = self.producer.get_info(self.path)["fps"]
        except Exception as e:
            raise ValueError(f"Error getting fps: {e}")

        interval_count = 0

        for frame in self.producer.read(self.path):
            self.frame_tracker.add_frame(frame)
            interval_count += 1
            if interval_count >= fps:
                interval_count = 0
                self.frame_tracker.increase_current_time()
            self.current_frame += 1
            yield frame, self.current_frame

    def play_realtime(self):
        """
        Starts the appropriate streaming process based on the frame count.
        """
        if not isinstance(self.frame_tracker, BaseManager):
            raise TypeError("frame_tracker must be a BaseManager")

        try:
            frame_count = self.producer.get_info(self.path)["frame_count"]
        except Exception as e:
            raise ValueError(f"Error getting frame count: {e}")

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

    @staticmethod
    def video_stream(frame_tracker, producer, path):
        """
        Handles streaming for video files. Frames are processed at a rate determined by the video's FPS.
        """
        try:
            fps = producer.get_info(path)["fps"]
        except Exception as e:
            raise ValueError(f"Error getting video information: {e}")

        base_interval = 1 / fps
        start_time = time.time()
        interval_count = 0

        for idx, frame in enumerate(producer.read(path)):
            target_time = start_time + (idx * base_interval)
            time.sleep(max(0, target_time - time.time()))

            frame_tracker.add_frame(frame)
            interval_count += 1
            if interval_count >= fps:
                interval_count = 0
                frame_tracker.increase_current_time()

    @staticmethod
    def normal_stream(frame_tracker, producer, path):
        """
        Handles streaming for non-video files. Frames are processed at regular intervals.
        """
        timer = Timer(interval=1)
        for frame in producer.read(path):
            if timer.is_time():
                frame_tracker.increase_current_time()
            frame_tracker.add_frame(frame)
