import av
import os
import cv2

from .producer import OpenCVProducer, PyAVProducer
from .player import Player
from .model import RecordMode


class Recorder:
    def __init__(self, player: Player, recording_path: str):
        self.record_mode = RecordMode.NONE
        self.player = player
        self.width = player.producer.width
        self.height = player.producer.height
        self.fps = player.play_fps
        self.recording_path = self._ensure_mp4_extension(recording_path)
        self._initialize_writer()

    def _ensure_mp4_extension(self, path):
        if not path.endswith(".mp4"):
            path += ".mp4"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _initialize_writer(self):
        if isinstance(self.player.producer, OpenCVProducer):
            self._init_opencv_writer()
        elif isinstance(self.player.producer, PyAVProducer):
            self._init_pyav_writer()

    def _init_opencv_writer(self):
        self.record_mode = RecordMode.OPENCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.recording_path, fourcc, self.fps, (self.width, self.height)
        )

    def _init_pyav_writer(self):
        self.record_mode = RecordMode.PYAV
        try:
            container = av.open(self.recording_path, mode="w")
        except av.AVError as e:
            # Handle AVError if necessary
            raise e
        stream = container.add_stream("h264", rate=self.fps)
        stream.width, stream.height = (self.width, self.height)
        stream.pix_fmt = "yuv420p"
        self.container = container
        self.stream = stream

    def add_frame(self, frame):
        try:
            if self.record_mode == RecordMode.OPENCV:
                self.writer.write(frame)
            elif self.record_mode == RecordMode.PYAV:
                self._add_frame_pyav(frame)
        except Exception as e:
            # Handle exceptions related to frame addition
            raise e

    def _add_frame_pyav(self, frame):
        frame = av.VideoFrame.from_ndarray(frame, format=self.player.producer.format)
        frame.pict_type = "NONE"
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        try:
            if self.record_mode == RecordMode.OPENCV:
                self.writer.release()
            elif self.record_mode == RecordMode.PYAV:
                self.container.close()
        except Exception as e:
            # Handle exceptions during closing
            raise e
