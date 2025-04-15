from typing import Any
import time

from stream_infer import Inference, Player
from stream_infer.algo import BaseAlgo
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.producer import OpenCVProducer

last_result_count = 0  # avoid duplicate print


class DebugAlgo(BaseAlgo):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.counter = 0

    def init(self):
        print(f"[DebugAlgo] Initialized")
        return True

    def run(self, frames):
        self.counter += 1
        result = {"counter": self.counter, "frames_count": len(frames) if frames else 0}
        frame_indices = [frame[0] for frame in frames]
        print(
            f"[DebugAlgo] Run: counter={self.counter}, frames_count={len(frames) if frames else 0}, frame_indices={frame_indices}"
        )
        return result


class DebugTimeConsumer(BaseAlgo):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.counter = 0

    def init(self):
        print(f"[DebugTimeConsumer] Initialized")
        return True

    def run(self, frames):
        self.counter += 1
        time.sleep(5)
        result = {"counter": self.counter}
        print(f"[DebugTimeConsumer] Run: counter={self.counter}")
        return result


class DebugDispatcher(DevelopDispatcher):
    def add_frame(self, frame: Any) -> None:
        frame = (self.current_frame_index, frame)
        super().add_frame(frame)


dispatcher = DebugDispatcher.create(mode="offline", buffer=30, logging_level="WARNING")
player = Player(dispatcher, OpenCVProducer(1920, 1080), source="./classroom.mp4")
inference = Inference(dispatcher, player)


@inference.process
def offline_process(inference: Inference, *args, **kwargs):
    global last_result_count
    data = inference.dispatcher.get_result("DebugAlgo", clear=False)
    if data is not None and len(data) > last_result_count:
        print(f"[Debug] get_result: {data}")
        print(f"[Debug] current_algo_names: {kwargs['current_algo_names']}")
        print(f"[Debug] last_algo_name: {kwargs['last_algo_name']}")
    last_result_count = len(data) if data else 0


inference.load_algo(DebugAlgo(), frame_count=10, frame_step=3, interval=1)
inference.load_algo(DebugTimeConsumer(), frame_count=10, frame_step=3, interval=5)
inference.start(fps=30, position=10)
