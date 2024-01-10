import cv2
from stream_infer import Inference, StreamlitApp
from stream_infer.dispatcher import DevelopDispatcher
from algos import YoloDetectionAlgo, PoseDetectionAlgo

dispatcher = DevelopDispatcher.create(mode="offline", buffer=5)
inference = Inference(dispatcher)
inference.load_algo(YoloDetectionAlgo("things"), 1, 0, 1)
inference.load_algo(PoseDetectionAlgo("pose"), 1, 0, 0.1)
app = StreamlitApp(inference)


# Set frame annotation func
@app.annotate_frame
def annotate_frame(app: StreamlitApp, name, data, frame):
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


# Set output display func
@app.output
def output(app: StreamlitApp, name, position, data):
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


app.start(producer_type="pyav", clear=False)  # options: opencv, pyav
