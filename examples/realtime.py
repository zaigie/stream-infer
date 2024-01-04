from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.algo import BaseAlgo
from stream_infer.producer import OpenCVProducer
from stream_infer.log import logger
from stream_infer.util import trans_position2time

import os

os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO

INFER_FRAME_WIDTH = 1920
INFER_FRAME_HEIGHT = 1080
FPS = 30


class YoloDectionAlgo(BaseAlgo):
    def init(self):
        self.model = YOLO("yolov8n.pt")

    def run(self, frames):
        try:
            result = self.model(frames[0])
            return [result[0].names[box.cls[0].int().item()] for box in result[0].boxes]
        except Exception as e:
            logger.error(e)
            return None


def realtime_process(inference: Inference, *args, **kwargs):
    current_time, data = inference.dispatcher.get_last_result(
        YoloDectionAlgo.__name__, clear=True
    )
    if data is not None:
        logger.debug(f"{trans_position2time(current_time)}: {data}")


if __name__ == "__main__":
    dispatcher = DevelopDispatcher.create()
    player = Player(
        dispatcher,
        OpenCVProducer(INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT),
        path="./classroom.mp4",
    )

    inference = Inference(dispatcher)
    inference.load_algo(YoloDectionAlgo(), frame_count=1, frame_step=1, interval=1)
    inference.set_custom_process(realtime_process)
    inference.start(player, fps=FPS)
