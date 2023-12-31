# Stream Infer

<p align="left">
   <strong>English</strong> | <a href="./README.zh.md">简体中文</a>
</p>

Stream Infer is a Python library designed for stream inference in video processing applications. It includes modular components for video frame generation, inference algorithms, and result export.

## Installation

```bash
pip install -U stream-infer
```

## Quick Start

Below is a simple example of Stream Infer to help you get started and understand what Stream Infer does.

This example uses the posture model of YOLOv8 for detection and draws the result into the cv2 window

> Due to the problem of Python multi-threading, real-time reasoning can not be displayed to the window for the time being

https://github.com/zaigie/stream_infer/assets/17232619/32aef0c9-89c7-4bc8-9dd6-25035bee2074

Video files in [sample-videos](https://github.com/intel-iot-devkit/sample-videos)

> You may need to install additional packages via pip to use this example:
>
> `pip install ultralytics supervision`

```python
from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher, DispatcherManager
from stream_infer.algo import BaseAlgo
from stream_infer.producer import PyAVProducer, OpenCVProducer
from stream_infer.log import logger

import cv2
import os

os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO
import supervision as sv

INFER_FRAME_WIDTH = 1920
INFER_FRAME_HEIGHT = 1080
PLAY_FPS = 30
OFFLINE = True


class YoloDectionAlgo(BaseAlgo):
    def init(self):
        self.model = YOLO("yolov8n.pt")

    def run(self, frames):
        # logger.debug(f"{self.name} starts running with {len(frames)} frames")
        try:
            result = self.model(frames[0])
            # logger.debug(f"{self.name} inference finished: {result[0]}")
            return result[0]
        except Exception as e:
            logger.error(e)
            return None


def annotate(frame, data):
    detections = sv.Detections.from_ultralytics(data)
    boundingbox_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    labels = [data.names[class_id] for class_id in detections.class_id]
    annotated_image = boundingbox_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )
    return annotated_image


if __name__ == "__main__":
    producer = OpenCVProducer(INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT)
    video_path = "./classroom.mp4"

    dispatcher = (
        DevelopDispatcher()
        if OFFLINE
        else DispatcherManager(DevelopDispatcher).create()
    )
    inference = Inference(dispatcher)
    inference.load_algo(YoloDectionAlgo(), frame_count=1, frame_step=1, interval=0.1)

    player = Player(dispatcher, producer, path=video_path)

    if OFFLINE:
        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
        for frame, current_frame in player.play(PLAY_FPS, position=0):
            inference.auto_run_specific(player.play_fps, current_frame)
            _, data = dispatcher.get_last_result(YoloDectionAlgo.__name__, clear=False)
            if data is None:
                continue

            frame = annotate(frame, data)
            cv2.imshow("Inference", frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
    else:
        player.play_async(PLAY_FPS)
        inference.run_async()
        while player.is_active():
            current_time, data = dispatcher.get_last_result(
                YoloDectionAlgo.__name__, clear=True
            )
            if data is None:
                continue
            logger.debug(f"{current_time} result: {data}")
        inference.stop()
        player.stop()
    dispatcher.clear()
```

## Features and Concepts

### Real-time Inference

![Sequence](./docs/img/real-time.png)

Real-time inference refers to inputting a video or stream, which plays at the normal real-time playback speed, adding frames to the track. The playback and inference processes are independent. Due to the time taken by inference, it results in varying delays, but with a reasonable frame rate set, it will not cause memory leaks or accumulation.

Real-time inference is more commonly applied in scenarios such as:

- Various live broadcast scenarios
- Real-time monitoring
- Real-time meetings
- Clinical surgeries
- ...

### Offline Inference

**Good Processing Performance**

![](./docs/img/offline_good.png)

**Poor Processing Performance**

![](./docs/img/offline_poor.png)

Offline inference refers to inputting a video (streams are not applicable here) and performing inference in parallel with frame fetching at the speed the computer can handle. Depending on machine performance, the total runtime may be longer or shorter than the video duration.

Offline inference is applied in **all non-real-time necessary** video structure analysis, such as:

- Post-meeting video analysis
- Surgical video replay
- ...

**Also, since the video reading and algorithm inference in offline inference run in sequence, it can be used to test algorithm performance and effects (as in the [Quick Start](#quick-start), displaying the video and algorithm data after inference through cv2), while real-time inference is not suitable for the algorithm development stage.**

## Modules

![Flowchart](./docs/img/flow.svg)

### BaseAlgo

We simply encapsulate all algorithms into classes with two functions: `init()` and `run()`, which is BaseAlgo.

Even though Stream Infer provides a framework about stream inference, **the actual algorithm functionality still needs to be written by you**. After writing, inherit the BaseAlgo class for unified encapsulation and calling.

For example, you have completed a head detection algorithm, and the inference call is:

```python
# https://modelscope.cn/models/damo/cv_tinynas_head-detection_damoyolo/summary
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_tinynas_head-detection_damoyolo'
input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'

head_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
result = head_detection(input_location)
print("result is : ", result)
```

Then, to perform stream inference of this algorithm in videos and streaming media, encapsulate it like this:

```python
from stream_infer.algo import BaseAlgo

class HeadDetectionAlgo(BaseAlgo):
    def init(self):
        self.model_id = 'damo/cv_tinynas_head-detection_damoyolo'
        self.head_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

    def run(self, frames):
        return self.head_detection(frames)
```

In this way, you have completed the encapsulation and will be able to call it normally in the future.

> [!CAUTION]
> In many cases, we use CUDA or MPS to accelerate inference, but please note that when you use either of these accelerations:
>
> **No Tensors should be returned** in the ` run()` function of `BaseAlgo` you inherit from! Please try to manually convert them into standard Python data formats, such as dictionaries.
>
> Or copy the tensor (to the CPU) and share it across multiple processes.
>
> This is due to the multi-process environment, and it may also be because my learning is not deep enough. If there are better solutions, I will try to resolve them.

### Dispatcher

Dispatcher serves as the central service linking playback and inference, caching inference frames, distributing inference frames, and collecting inference time and result data.

Dispatcher provides functions for adding/getting frames, adding/getting inference results and times. You don't need to worry about others, but to be able to get the results and print them conveniently, store them in other locations, you need to focus on the `collect_result()` function.

Here is their source code implementation:

```python
def collect_result(self, inference_result):
    if inference_result is not None:
        time = str(inference_result[0])
        name = inference_result[1]
        data = inference_result[2]
        if self.collect_results.get(name) is None:
            self.collect_results[name] = {}
        self.collect_results[name][time] = data
```

The format of the collected `collect_results` is roughly as follows:

```json
{
  "HeadDetectionAlgo": {
    "1": { "scores": [], "boxes": [] },
    "2": { "scores": [], "boxes": [] }
  },
  "other": {
    "60": { "a": 1 },
    "120": { "a": 2 }
  }
}
```

On this basis, if you want to request the result to a REST service, or do other operations on the existing data before requesting, it can be achieved by **inheriting the Dispatcher class** and rewriting the function:

```python
from stream_infer.dispatcher import Dispatcher, DispatcherManager
import requests
...
class SelfDispatcher(Dispatcher):
    def __init__(self, max_size: int = 120):
        super().__init__(max_size)
        self.sess = requests.Session()
        ...

    def collect_result(self, inference_result):
        super().__init__(inference_result)
        req_data = {
            "time" = inference_result[0]
            "name" = inference_result[1]
            "data" = inference_result[2]
        }
        self.sess.post("http://xxx.com/result/", json=req_data)
...

# In Offline
dispatcher = SelfDispatcher()

# In Real-Time
dispatcher = DispatcherManager(SelfDispatcher).create(max_size=150)
```

> You may have noticed that the way to instantiate the dispatcher is different in offline and real-time environments. This is because **in real-time environments, playback and inference are not in the same process**. Both need to share the same dispatcher, so DispatcherManager proxy is used.

### Producer

Producer loads videos or streaming media in different ways, such as PyAV, OpenCV, ImageIO (only applicable offline), etc., and adjusts or transforms the width, height, and color space of the frames, finally returning each frame as a numpy array.

Instantiating a Producer often requires inputting the frame width and height needed for inference and the color order. The default color order is the same as the BGR order returned by `cv2.imread()`.

```python
from stream_infer.producer import PyAVProducer, OpenCVProducer

producer = PyAVProducer(1920, 1080)
producer = OpenCVProducer(1920, 1080)
```

### Inference

Inference is the core of the framework, implementing functions such as loading algorithms and running inference.

An Inference object needs to input a Dispatcher object for frame fetching and sending inference results, etc.

```python
from stream_infer import Inference

inference = Inference(dispatcher)
```

When you need to load an algorithm, here is an example using the [BaseAlgo](#basealgo) above:

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(dispatcher)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

The parameters for loading the algorithm are the core features of the framework, allowing you to freely implement the frame fetching logic:

- frame_count: The number of frames the algorithm needs to fetch, which is the number of frames finally received in the run() function.
- frame_step: Fetch 1 frame every `frame_step`, a total of `frame_count` frames. If this parameter is filled with fps, it means fetching the last `frame_count` frames per second.
- interval: In seconds, it represents the frequency of algorithm calls, such as `AnyOtherAlgo` will only be called once a minute, saving resources when it is not necessary to call it.

### Player

Player inputs dispatcher, producer, and video/streaming media address for playback and inference.

```python
from stream_infer import Player

player = Player(dispatcher, producer, video_path)
```

Player has two functions to execute in offline and real-time inference modes, respectively:

```python
player.play(fps=None, position=0)
player.play_async(fps=None)
```

Both functions can input an fps parameter, which represents the playback frame rate here. **If the frame rate of the video source is higher than this number, frames will be skipped to force playback at this specified frame rate.** This can also save performance to some extent.

In the offline, you can also specify the position of the playback, the position parameter will receive a parameter in seconds.

### Play & Run

#### Offline Running

Player's `play()` returns an iterable object, and calling `inference.auto_run_specific()` in the loop will automatically determine which algorithm to run based on the current frame index:

```python
if __name__ == "__main__":
    ...
    for frame, current_frame in player.play(PLAY_FPS):
        current_algo_name = inference.auto_run_specific(
            player.play_fps, current_frame
        )
        # Other operations, such as drawing the picture window
        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
        cv2.imshow("Inference", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
```

As described in [Offline Inference](#offline-inference), all the executions above are synchronized in one process and one thread, so you can take your time to complete the operations you want, such as algorithm effect verification (as in the [Quick Start](#quick-start), getting the inference result and displaying boxes to the window, etc.), even if it is stuttered due to synchronous operation, everything is accurate.

#### Streamlit Debug

A web application based on streamlit is provided, which facilitates development and debugging. You only need to inherit `StreamInferApp` and override two functions:

- annotate_frame(self, name, data, frame)
- output(name, position, data)

The former is used to customize the content drawn on the frame, and the latter is for customizing streamlit data display components (by default, it appends st.text()).

A simple example:

```python
import streamlit as st
import os
import cv2

from stream_infer import Inference, StreamInferApp
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.algo import BaseAlgo
from stream_infer.log import logger

os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO
import supervision as sv


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


class CustomStreamInferApp(StreamInferApp):
    def annotate_frame(self, name, data, frame):
        if name == "pose":
            keypoints = data.keypoints
            for person in keypoints.data:
                for kp in person:
                    x, y, conf = kp
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        else:
            detections = sv.Detections.from_ultralytics(data)
            boundingbox_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            labels = [data.names[class_id] for class_id in detections.class_id]
            frame = boundingbox_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )
        return frame


if __name__ == "__main__":
    dispatcher = DevelopDispatcher(150)
    inference = Inference(dispatcher)
    inference.load_algo(
        YoloDetectionAlgo("things"), frame_count=1, frame_step=30, interval=1
    )
    inference.load_algo(
        PoseDetectionAlgo("pose"), frame_count=1, frame_step=1, interval=0.1
    )
    app = CustomStreamInferApp(inference)
    app.start()
```

#### Real-time Running

Just run Player's `play_async()` and Inference's `run_async()`:

> It is particularly important to note that we recommend not exceeding 30 frames per second for the playback frame rate when running in real-time. Firstly, a high frame rate does not help much with the accuracy of analysis results. Secondly, it will lead to memory leaks and frame accumulation.

```python
if __name__ == "__main__":
    ...
    player.play_async(PLAY_FPS)
    inference.run_async()
    while player.is_active():
        pass
        # Other operations
    inference.stop()
    player.stop()
```

Monitor the playback status with `player.is_active()`, and manually end the inference thread and playback process after playback is complete.

## License

Stream Infer is licensed under the [Apache License](LICENSE).
