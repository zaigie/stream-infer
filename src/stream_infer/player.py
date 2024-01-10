import time
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.managers import BaseProxy
from typing import Union

from .dispatcher import Dispatcher
from .producer import OpenCVProducer, PyAVProducer
from .util import position2time
from .log import logger


class Player:
    def __init__(
        self,
        dispatcher: Union[Dispatcher, BaseProxy],
        producer: Union[OpenCVProducer, PyAVProducer],
        source: Union[str, int],
        show_progress: bool = True,
    ):
        self.dispatcher = dispatcher
        self.producer = producer
        self.source = source
        self.show_progress = show_progress
        try:
            self.info = self.producer.get_info(self.source)
        except Exception as e:
            raise ValueError(f"Error getting info: {e}")
        self.fps = self.info["fps"]
        self.play_fps = self.fps
        self.frame_count = self.info["frame_count"]
        self.process = None
        self.is_end = mp.Value("b", False)

    def play(self, fps=None, position=0):
        fps = self.fps if fps is None else fps
        self.play_fps = fps
        interval_count = 0
        if self.show_progress:
            pbar = tqdm(
                total=self.info["total_seconds"],
                desc="Video Time",
                leave=True,
                unit="sec",
            )

        if position > 0:
            self.dispatcher.set_current_position(position)
            self.dispatcher.set_current_frame_index(fps * position)
            if self.show_progress:
                pbar.update(fps * position)

        for frame in self.producer.read(self.source, fps, position):
            self.dispatcher.add_frame(frame)
            interval_count += 1
            if interval_count >= fps:
                interval_count = 0
                self.dispatcher.increase_current_position()
                if self.show_progress:
                    pbar.update(1)
                else:
                    logger.debug(
                        f"{self.get_play_time()}/{position2time(self.info['total_seconds'])}"
                    )
            yield frame, self.dispatcher.get_current_frame_index()
        if self.show_progress:
            pbar.close()

    def play_async(self, fps=None):
        """
        Starts the appropriate streaming process based on the frame count.
        """
        if not isinstance(self.dispatcher, BaseProxy):
            logger.error(
                f"Dispatcher is not an proxy: {type(self.dispatcher)}, use create(mode='realtime') to create"
            )
            raise ValueError(
                f"Dispatcher is not an proxy: {type(self.dispatcher)}, use create(mode='realtime') to create"
            )

        if fps is None or fps >= self.fps:
            fps = self.fps
            if fps > 30:
                logger.warning(
                    f"FPS {fps} is too high, if your player is playing more slowly than the actual time, set a lower play fps"
                )
        self.play_fps = fps

        if self.frame_count <= 0:
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
        return position2time(self.dispatcher.get_current_position())

    def video_stream(self):
        """
        Handles streaming for video files. Frames are processed at a rate determined by the video's FPS.
        """
        base_interval = 1 / self.play_fps
        start_time = time.time()
        interval_count = 0
        if self.show_progress:
            pbar = tqdm(
                total=self.info["total_seconds"],
                desc="Streaming Video Time",
                leave=True,
                unit="sec",
            )
        for idx, frame in enumerate(self.producer.read(self.source, self.play_fps)):
            target_time = start_time + (idx * base_interval)
            time.sleep(max(0, target_time - time.time()))
            self.dispatcher.add_frame(frame)
            interval_count += 1
            if interval_count >= self.play_fps:
                interval_count = 0
                self.dispatcher.increase_current_position()
                if self.show_progress:
                    pbar.update(1)
                else:
                    logger.debug(
                        f"{self.get_play_time()}/{position2time(self.info['total_seconds'])}"
                    )
        if self.show_progress:
            pbar.close()
        self.is_end.value = True

    def normal_stream(self):
        """
        Handles streaming for non-video files. Frames are processed at regular intervals.
        """
        for frame in self.producer.read(self.source, self.play_fps):
            if self.dispatcher.get_current_frame_index() % self.play_fps == 0:
                self.dispatcher.increase_current_position()
                logger.debug(f"{self.get_play_time()}")
            self.dispatcher.add_frame(frame)

        self.is_end.value = True
