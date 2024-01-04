from stream_infer import Inference, StreamInferApp
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.algo import BaseAlgo
from stream_infer.log import logger

import os
import cv2

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


def annotate_frame(app: StreamInferApp, name, data, frame):
    if name == "pose":
        keypoints = data.keypoints
        for person in keypoints.data:
            for kp in person:
                x, y, conf = kp
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    elif name == "things":
        names = data.names
        boxes = data.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            name = names[box.cls[0].int().item()]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    return frame


def output(app: StreamInferApp, name, position, data):
    if data is None:
        return
    things = [data.names[box.cls[0].int().item()] for box in data.boxes]

    def count_things(name):
        count = 0
        for thing in things:
            if thing == name:
                count += 1
        return count

    if name == "things":
        types = set(things)
        cols = app.output_widgets[name].columns(len(types))
        for i, t in enumerate(types):
            cols[i].metric(t, count_things(t))

    if name == "pose":
        app.output_widgets[name] = app.output_widgets[name].container()
        app.output_widgets[name].text(f"{position}: {things}")


if __name__ == "__main__":
    dispatcher = DevelopDispatcher.create(max_size=5, offline=True)
    inference = Inference(dispatcher)
    inference.load_algo(
        YoloDetectionAlgo("things"), frame_count=1, frame_step=0, interval=1
    )
    inference.load_algo(
        PoseDetectionAlgo("pose"), frame_count=1, frame_step=0, interval=0.1
    )
    app = StreamInferApp(inference)
    app.set_annotate_frame(annotate_frame)
    app.set_output(output)
    app.start(use_opencv=True, clear=False)  # use_opencv for use OpenCVProducer
