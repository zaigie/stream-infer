import cv2
from algos import YoloDetectionAlgo

from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.producer import OpenCVProducer

dispatcher = DevelopDispatcher.create(mode="offline", buffer=1)
inference = Inference(dispatcher)


# Set process func in offline inference
@inference.process
def offline_process(inference: Inference, *args, **kwargs):
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
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    frame = kwargs.get("frame")
    # current_algo_names = kwargs.get("current_algo_names")
    _, data = inference.dispatcher.get_last_result("YoloDetectionAlgo", clear=False)
    if data is not None:
        draw_boxes(frame, data)
        cv2.imshow("Inference", frame)
        cv2.waitKey(1)


player = Player(dispatcher, OpenCVProducer(1920, 1080), source="./classroom.mp4")
inference.load_algo(YoloDetectionAlgo(), frame_count=1, frame_step=0, interval=0.1)
cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
inference.start(
    player, fps=30, position=0, mode="offline", recording_path="./processed.mp4"
)
cv2.destroyAllWindows()
