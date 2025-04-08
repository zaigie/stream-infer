from typing import Any, List

import cv2
import numpy as np

from stream_infer import Inference, Player
from stream_infer.algo import BaseAlgo
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.producer import OpenCVProducer

last_result_count = 0


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

        print(
            f"[DebugAlgo] Run: counter={self.counter}, frames_count={len(frames) if frames else 0}"
        )

        return result


class DebugDispatcher(DevelopDispatcher):
    def get_frames(self, count: int, step: int) -> List[Any]:
        frames = super().get_frames(count, step)

        if frames and len(frames) > 0:
            frame_indices = []
            for i, frame in enumerate(frames):
                try:
                    top_left = frame[10:60, 10:100]
                    top_right = frame[10:60, frame.shape[1] - 100 : frame.shape[1] - 10]
                    bottom_left = frame[
                        frame.shape[0] - 60 : frame.shape[0] - 10, 10:100
                    ]
                    bottom_right = frame[
                        frame.shape[0] - 60 : frame.shape[0] - 10,
                        frame.shape[1] - 100 : frame.shape[1] - 10,
                    ]

                    green_pixels = []
                    for roi in [top_left, top_right, bottom_left, bottom_right]:
                        green_mask = (
                            (roi[:, :, 1] > 200)
                            & (roi[:, :, 0] < 100)
                            & (roi[:, :, 2] < 100)
                        )
                        green_pixels.append(np.sum(green_mask))

                    if max(green_pixels) > 0:
                        estimated_index = self.current_frame_index - len(self.queue) + i
                        frame_indices.append(f"{estimated_index}")
                    else:
                        frame_indices.append("null")
                except Exception as e:
                    frame_indices.append(f"extract failed: {str(e)}")

            print(
                f"[Debug] get_frames: count={count}, frame_index={self.current_frame_index}, queue_len={len(self.queue)}, frame_indices={frame_indices}"
            )

        return frames


class DebugProducer(OpenCVProducer):
    def __init__(self, width: int, height: int, cvt_code=None):
        super().__init__(width, height, cvt_code)
        self.frame_indices = {}

    def read(self, source, fps=None, position=0):
        # print(f"[Debug] read: source={source}, fps={fps}, position={position}")
        frame_index = 0
        for frame in super().read(source, fps, position):
            colored_frame = frame.copy()
            cv2.putText(
                colored_frame,
                f"Frame: {frame_index}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                colored_frame,
                f"{frame_index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                colored_frame,
                f"{frame_index}",
                (colored_frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                colored_frame,
                f"{frame_index}",
                (10, colored_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                colored_frame,
                f"{frame_index}",
                (colored_frame.shape[1] - 100, colored_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            frame_index += 1
            yield colored_frame


dispatcher = DebugDispatcher.create(mode="offline", buffer=5)
inference = Inference(dispatcher)


@inference.process
def offline_process(inference: Inference, *args, **kwargs):
    global last_result_count
    data = inference.dispatcher.get_result("DebugAlgo", clear=False)
    if data is not None and len(data) > last_result_count:
        print(f"[Debug] get_result: {data}")
    last_result_count = len(data) if data else 0


inference.load_algo(DebugAlgo(), frame_count=10, frame_step=0, interval=1)
player = Player(
    dispatcher, DebugProducer(1920, 1080), source="./classroom.mp4", show_progress=False
)
inference.start(player, fps=30, position=0, mode="offline", logging_level="WARNING")
