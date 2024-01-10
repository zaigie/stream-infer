import os

from stream_infer.algo import BaseAlgo
from stream_infer.log import logger

os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO


class YoloDetectionAlgo(BaseAlgo):
    def init(self):
        self.model = YOLO("yolov8n.pt")

    def run(self, frames):
        try:
            result = self.model(frames[0])
            return result[0]
        except Exception as e:
            logger.error(e)
            return None


class YoloDetectionAlgo2(BaseAlgo):
    def init(self):
        self.model = YOLO("yolov8n.pt")

    def run(self, frames):
        try:
            result = self.model(frames[0])
            return [result[0].names[box.cls[0].int().item()] for box in result[0].boxes]
        except Exception as e:
            logger.error(e)
            return None


class PoseDetectionAlgo(BaseAlgo):
    def init(self):
        self.model = YOLO("yolov8n-pose.pt")

    def run(self, frames):
        try:
            result = self.model(frames[0])
            return result[0]
        except Exception as e:
            logger.error(e)
            return None
