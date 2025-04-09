import cv2
import streamlit as st
from algos import PoseDetectionAlgo, YoloDetectionAlgo

from stream_infer import Inference, StreamlitApp
from stream_infer.dispatcher import DevelopDispatcher

dispatcher = DevelopDispatcher.create(mode="offline", buffer=5)
inference = Inference(dispatcher)  # Player will be created automatically
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


algo_containers = {}
algo_states = {}


# Set output display func
@app.output
def output(app: StreamlitApp, name, position, data):
    if data is None:
        return

    global algo_containers, algo_states
    if name not in algo_containers:
        algo_containers[name] = app.output_widgets[name].empty()
        algo_states[name] = {"last_had_data": False}

    algo_containers[name].empty()

    container = algo_containers[name].container()

    if name == "things":
        import pandas as pd

        things = [data.names[box.cls[0].int().item()] for box in data.boxes]

        if not things:
            container.subheader("未检测到物体")
            df = pd.DataFrame({"物体类型": [], "数量": []})
            algo_states[name]["last_had_data"] = False
        else:
            thing_counts = {}
            for thing in things:
                if thing in thing_counts:
                    thing_counts[thing] += 1
                else:
                    thing_counts[thing] = 1

            container.subheader(f"检测到 {len(things)} 个物体")
            df = pd.DataFrame(
                {
                    "物体类型": list(thing_counts.keys()),
                    "数量": list(thing_counts.values()),
                }
            )
            algo_states[name]["last_had_data"] = True

        container.dataframe(
            df,
            column_config={
                "物体类型": st.column_config.TextColumn("物体类型"),
                "数量": st.column_config.NumberColumn("数量", format="%d"),
            },
            hide_index=True,
            use_container_width=True,
        )

    elif name == "pose":
        keypoints = data.keypoints
        num_persons = len(keypoints.data) if hasattr(keypoints, "data") else 0

        summary = f"时间: {position}秒 | 检测到 {num_persons} 人"
        container.subheader(summary)

        if num_persons == 0:
            algo_states[name]["last_had_data"] = False
            return

        if num_persons > 0 and num_persons <= 3:
            for i, person in enumerate(keypoints.data[:3]):
                with container.expander(f"人物 #{i+1} 详情", expanded=False):
                    valid_kps = sum(1 for kp in person if kp[2] > 0.5)
                    st.text(f"有效关键点: {valid_kps}/{len(person)}")

        algo_states[name]["last_had_data"] = num_persons > 0


app.start(producer_type="pyav", clear=False)  # options: opencv, pyav
