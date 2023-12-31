from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.algo import BaseAlgo
from stream_infer.producer import OpenCVProducer
from stream_infer.log import logger

import os
import cv2

os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO

INFER_FRAME_WIDTH = 1920
INFER_FRAME_HEIGHT = 1080
PLAY_FPS = 30


class YoloDectionAlgo(BaseAlgo):
    def init(self):
        self.model = YOLO("yolov8n.pt")

    def run(self, frames):
        try:
            result = self.model(frames[0])
            return result[0]
        except Exception as e:
            logger.error(e)
            return None


def draw_boxes(frame, data):
    names = data.names
    boxes = data.boxes
    for i in range(len(boxes)):
        box = boxes[i]
        name = names[box.cls[0].int().item()]
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )


def offline_progress(inference: Inference, *args, **kwargs):
    frame = kwargs.get("frame")
    _, data = inference.dispatcher.get_last_result(
        YoloDectionAlgo.__name__, clear=False
    )
    if data is not None:
        draw_boxes(frame, data)
        cv2.imshow("Inference", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    producer = OpenCVProducer(INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT)
    video_path = "./classroom.mp4"
    dispatcher = DevelopDispatcher()
    player = Player(dispatcher, producer, path=video_path)

    inference = Inference(dispatcher)
    inference.load_algo(YoloDectionAlgo(), frame_count=1, frame_step=1, interval=0.1)
    inference.set_custom_progress(offline_progress)
    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    inference.start(player, fps=PLAY_FPS, position=0)
    cv2.destroyAllWindows()
