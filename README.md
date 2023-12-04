# Stream Infer

<p align="left">
   <strong>English</strong> | <a href="./README.zh.md">简体中文</a>
</p>

Stream Infer is a Python library designed for stream inference in video processing applications. It contains modular components for video frame generation, inference algorithms, and result exportation.

## Installation

```bash
pip install stream-infer
```

## Quick Start

Here is a simple example of Stream Infer to help you get started:

```python
from stream_infer import Inference, FrameTracker, TrackerManager, Player
from stream_infer.algo import BaseAlgo
from stream_infer.exporter import BaseExporter
from stream_infer.producer import PyAVProducer
from stream_infer.log import logger

import time

class ExampleAlgo(BaseAlgo):
    def init(self):
        logger.info(f"{self.name} initialized")

    def run(self, frames):
        logger.debug(f"{self.name} starts inferring {len(frames)} frames")
        time.sleep(0.3)
        result = {"name": self.name}
        logger.debug(f"{self.name} inference completed: {result}")
        return result

class Exporter(BaseExporter):
    def send(self):
        if len(self.results) == 0:
            return
        logger.debug(self.results[-1])

INFER_FRAME_WIDTH = 1920
INFER_FRAME_HEIGHT = 1080
OFFLINE = False

video = "/path/to/your/video.mp4"
play_fps = 30

producer = PyAVProducer(INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT)
exporter = Exporter()

if __name__ == "__main__":
    frame_tracker = FrameTracker() if OFFLINE else TrackerManager().create()

    inference = Inference(frame_tracker, exporter)
    inference.load_algo(ExampleAlgo(), frame_count=1, frame_step=play_fps, interval=1)
    inference.load_algo(ExampleAlgo("emotion"), 5, 6, 60)

    player = Player(producer, frame_tracker, video)
    inference.start(player, fps=play_fps, is_offline=OFFLINE)
```

## Features and Concepts

### Real-time Inference

Real-time inference refers to inputting a video or stream, where the video or stream plays at normal real-time speed, adding frames to the frame track. The playback process and the inference process are independent. Since inference will take a certain amount of time, it will cause more or less delay in results, but it will not cause memory leakage or pile-up.

### Offline Inference

Offline inference refers to inputting a video (streams are not applicable here), processing at the speed the current computer can handle. Frame fetching and inference are interleaved. Since inference will take a certain amount of time, depending on the machine performance, the total runtime of the process may be longer or shorter than the video duration.

## Modules

### Step1. BaseAlgo

We simply encapsulate and abstract all algorithms into classes with `init()` and `run()` functions, which is what BaseAlgo is.

Even though Stream Infer provides a framework about stream inference, the actual algorithm still needs to be written by yourself. After writing it, inherit the BaseAlgo class for unified encapsulation and calling.

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

To perform stream inference of this algorithm in videos and streaming media, encapsulate it like this:

```python
from stream_infer.algo import BaseAlgo

class HeadDetectionAlgo(BaseAlgo):
    def init(self):
        self.model_id = 'damo/cv_tinynas_head-detection_damoyolo'
        self.head_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

    def run(self, frames):
        return self.head_detection(frames)
```

This way, you have completed the encapsulation and can normally call it in the future.

### Step2. Exporter

Exporter is used to collect inference results, making it easy to collect data such as algorithm names, results, and times.

Currently, only BaseExporter is implemented. More like RedisExporter, TDEngineExporter, MySQLExporter, etc., will be developed soon.

Here is the simple source code of BaseExporter and an implementation called PrintExporter:

```python
class BaseExporter:
    def __init__(self):
        self.results = []

    def collect(self, inference_result):
        if inference_result is not None:
            self.results.append(inference_result)

    def send(self, *args, **kwargs):
        raise NotImplementedError

    def send_all(self):
        for result in self.results:
            self.send(result)

    def clear(self):
        self.results.clear()


class PrintExporter(BaseExporter):
    def send(self):
        if len(self.results) == 0:
            return
        print(self.results[-1])
```

After printing, you will get the last inference result. The structure of `inference_result` is `(current_time, algo_instance.name, result)`

### Step3. Producer

Producer loads videos or streaming media in different ways, such as PyAV, FFmpeg, ImageIO, etc., and adjusts or transforms the frames in terms of width, height, and color space, finally returning each frame as a numpy array.

Instantiating a Producer often requires inputting the frame width, height, and color space needed for inference, with color defaulting to BGR24.

```python
from stream_infer.producer import PyAVProducer

producer = PyAVProducer(1920, 1080)
```

### Step4. FrameTracker

FrameTracker can be understood as a frame track, supporting caching the latest frames into a queue, with a settable queue `max_size` (default is 120 frames).

At the same time, FrameTracker also assumes the responsibility of obtaining the current playback time based on fps and the current frame.

It should be noted that real-time and offline are two different operating modes. In real-time operation, the producer and inference are not in the same process and belong to a multi-process environment. Both the producer and inference need to access the same FrameTracker object. Therefore, when you need to run stream inference in real-time, you need to create a FrameTracker object through TrackerManager, rather than instantiating it directly through FrameTracker.

```python
from stream_infer import FrameTracker, TrackerManager

# In an offline environment
frame_tracker = FrameTracker()

# In a real-time environment
frame_tracker = TrackerManager().create(max_size=150)
```

### Step5. Inference

Inference is the core of this framework, implementing functions such as loading algorithms and running inference.

An Inference object requires input of the FrameTracker object (for frame fetching) and the Exporter object (for collection).

```python
from stream_infer import Inference
inference = Inference(frame_tracker, exporter)
```

When you need to load algorithms, here's an example from Step1:

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(frame_tracker, exporter)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

Here, we can specify a name for HeadDetectionAlgo, used to identify the running algorithm name (needed when collecting with Exporter and to avoid duplication). Also note the parameters:

- frame_count: The number of frames the algorithm needs to fetch, which is also the number of frames finally received in the `run()` function.
- frame_step: Indicates that for every `frame_step`, `frame_count` frames are fetched. If this parameter is filled with fps, it means that the last `frame_count` frames per second are fetched.
- interval: The frequency of algorithm invocation, like the above `AnyOtherAlgo` which will only be called once a minute.

### Step6. Player & run

Player inputs the producer, frame tracker, and video/streaming media address for playback and inference.

A Player requires the above parameters.

```python
from stream_infer import Player
player = Player(producer, frame_tracker, video)
```

Having loaded the algorithms, with a collector, frame track, and player, you can start running the inference.

In fact, you generally **do not** execute the code for running inference in the Quick Start in actual production development:

```python
# !!! Do not use in actual production development
inference.start(player, fps=play_fps, is_offline=OFFLINE)
```

It combines the offline and real-time running methods, but this is not quite correct, as we often need to debug or handle other tasks during the run. Let's take a look at the source code of the `start()` function.

```python
def start(self, player, fps: int = None, is_offline: bool = False):
    """
    Easy to use function to start inference with realtime mode.
    """
    if is_offline:
        for _, current_frame in player.play(fps):
            self.auto_run_specific_inference(player.fps, current_frame)
    else:
        player.play_realtime(fps)
        while player.is_active():
            self.run_inference()
```

From this, we can easily extract the key functions for real-time and offline operation.

#### Real-time Operation

```python
player.play_realtime(fps)
while player.is_active():
    print(player.get_current_time_str())
    self.run_inference()
    print(exporter.send())
    # Other action
```

#### Offline Operation

```python
for _, current_frame in player.play(fps):
    self.auto_run_specific_inference(player.fps, current_frame)
    print(exporter.send())
    # Other action
```

## License

Stream Infer is licensed under the [Apache License](LICENSE).
