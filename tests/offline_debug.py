from typing import Any, List

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
        current_idx = self.current_frame_index
        frame = (current_idx, frame)
        super().add_frame(frame)


dispatcher = DebugDispatcher.create(mode="offline", buffer=30)
inference = Inference(dispatcher)


@inference.process
def offline_process(inference: Inference, *args, **kwargs):
    global last_result_count
    data = inference.dispatcher.get_result("DebugAlgo", clear=False)
    if data is not None and len(data) > last_result_count:
        print(f"[Debug] get_result: {data}")
    last_result_count = len(data) if data else 0


inference.load_algo(DebugAlgo(), frame_count=10, frame_step=3, interval=1)
player = Player(
    dispatcher,
    OpenCVProducer(1920, 1080),
    source="./classroom.mp4",
    show_progress=False,
)
inference.start(player, fps=30, position=0, mode="offline", logging_level="WARNING")
