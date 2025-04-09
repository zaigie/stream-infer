from typing import Any

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


class DebugDispatcher(DevelopDispatcher):
    def add_frame(self, frame: Any) -> None:
        frame = (self.current_frame_index, frame)
        super().add_frame(frame)


dispatcher = DebugDispatcher.create(mode="offline", buffer=30, logging_level="WARNING")
inference = Inference(dispatcher)


@inference.process
def offline_process(inference: Inference, frame, current_algo_names):
    global last_result_count
    data = inference.dispatcher.get_result("DebugAlgo", clear=False)
    if data is not None and len(data) > last_result_count:
        print(f"[Debug] get_result: {data}")
    last_result_count = len(data) if data else 0


inference.load_algo(DebugAlgo(), frame_count=10, frame_step=3, interval=1)
inference.start(
    Player(
        dispatcher,
        OpenCVProducer(1920, 1080),
        source="./classroom.mp4",
        show_progress=False,
    ),
    fps=30,
    position=10,
)
