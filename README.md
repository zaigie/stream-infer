# <img src="./docs/img/logo_.png?raw=true" alt="Stream Infer" height="60px">

[![PyPI](https://img.shields.io/pypi/v/stream-infer?color=dark-green)](https://pypi.org/project/stream-infer/)
[![PyPI downloads](https://img.shields.io/pypi/dm/stream-infer?color=dark-green)](https://pypi.org/project/stream-infer/)
[![GitHub license](https://img.shields.io/github/license/zaigie/stream-infer?color=orange)](./LICENSE)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/zaigie/stream-infer)](https://github.com/zaigie/stream-infer/graphs/commit-activity)

<p align="left">
   <strong>English</strong> | <a href="./README-zh.md">简体中文</a>
</p>

Stream Infer is a Python library designed for streaming inference in video processing applications, enabling the integration of various image algorithms for video structuring. It supports both real-time and offline inference modes.

In short, Stream Infer is a device hardware and ML framework agnostic, supporting a lightweight version of [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) for cloud or edge IoT devices.

---

Often we want to use one or more image algorithms and models to analyze videos, and be able to set different frame capture logics and invocation frequencies for these algorithms, ultimately obtaining structured inferential results.

Sometimes it's even necessary to connect to a real-time camera or a live web stream, to infer and feedback results according to pre-set rules.

If you have the above requirements, then a simple Stream Infer can meet all your needs from development to debugging and to production operation.

![Flow](./docs/img/flow.svg?raw=true)

## Features

- [x] Minimal dependencies, purely written in Python, not limited by hardware architecture
- [x] Supports all Python-based algorithm deployment frameworks
- [x] Run a video inference task in less than 10 lines code
- [x] Supports both offline and real-time inference, just by changing one parameter
  - Offline inference completely traverses the video frame by frame and follows the preset algorithm and logical serial inference to get the result
  - Real-time inference separates frame fetching from inference and generates large or small delays depending on the performance of the processing device
- [x] Offline inference supports recording video files to a local disk
- [x] Supports visual development and debugging on local/remote servers via [streamlit](https://github.com/streamlit/streamlit)
- [x] Support parameterized dynamic call, easy inference server development
- [x] Modules are low-coupled, with clear division of labor
- [ ] Recording and streaming under real-time inference

## Installation

```bash
pip install -U stream-infer
```

## Quick Start

All examples depend on YOLOv8, and additional packages may be required for these examples:

```bash
pip install ultralytics
```

The video files used in the examples are available at [sample-videos](https://github.com/intel-iot-devkit/sample-videos)

### Offline Inference

https://github.com/zaigie/stream_infer/assets/17232619/32aef0c9-89c7-4bc8-9dd6-25035bee2074

Offline inference processes a **finite-length video or stream** at the speed the computer can handle, performing inference serially while capturing frames.

Since inference invariably takes time, depending on machine performance, the entire process's duration **may be longer or shorter than the video's length**.

Besides debugging during the development phase, offline inference can also be used for video structuring analysis in production environments where real-time processing is not essential, such as:

- Post-meeting video analysis
- Surgical video review
- ...

View and run the demo:

- General operation: [examples/offline_general.py](./examples/offline_general.py)
- Set up handlers to process frames and inference results, and display and record them using cv2: [examples/offline_custom_process_record.py](./examples/offline_custom_process_record.py)

> [!WARNING]
> This example `offline_custom_process_record.py` use OpenCV GUI-related features, such as presentation Windows, which can be manually installed either `opencv-python` or `opencv-contrib-python`, or simply
>
> `pip install -U stream-infer[desktop]`

### Real-time Inference

Real-time inference handles a **finite/infinite-length video or stream**, playing at normal speed if finite.

In this mode, a custom-size frame track is maintained, continuously adding the current frame to the track, with the **playback process independent of the inference process**.

Since inference takes time and **playback does not wait for inference results**, there will inevitably be a delay between the inference results and the current scene.

Real-time inference is not suitable for the development phase but is often used in production environments for real-time scenarios like RTMP/RTSP/camera feeds:

- Various live broadcast scenarios
- Real-time monitoring
- Live meetings
- Clinical surgeries
- ...

View and run the demo:

- General Operation: [examples/realtime_general.py](./examples/realtime_general.py)
- Set up the handler and manually print the inference result: [examples/realtime_custom_process.py](./examples/realtime_custom_process.py)

### Dynamic Execution

Leveraging Python's reflection and dynamic import capabilities, it supports configuring all the parameters required for inference tasks through JSON.

This mode is mainly useful for the development of inference servers, where structured data can be passed in via REST/gRPC or other methods to initiate an inference task.

View and run the demo: [examples/dynamic_app.py](./examples/dynamic_app.py)

### Visualization Development & Debugging

https://github.com/zaigie/stream_infer/assets/17232619/6cbd6858-0292-4759-8d4c-ace154567f8e

Implemented through a visual web application using [streamlit](https://github.com/streamlit/streamlit).

> The current interface text is Chinese.

This mode is primarily used for algorithm development and debugging on local/remote servers, supporting custom frame drawing and data display components.

To run this feature, install the server version:

```bash
pip install -U 'stream-infer[server]'
```

View and run the demo: [examples/streamlit_app.py](./examples/streamlit_app.py)

```bash
streamlit run streamlit_app.py
```

## Modules

Please read the following content in conjunction with [examples](./examples).

### BaseAlgo

Stream Infer simply encapsulates and abstracts all algorithms into classes with `init()` and `run()` functions, implementing BaseAlgo.

Although Stream Infer provides a framework for streaming inference, the actual algorithm functionality still needs to be developed by the user and **inherited from the BaseAlgo class** for uniform encapsulation and invocation.

For instance, if you have completed a real-time head detection algorithm, the official invocation method is:

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

To use it in Stream Infer, encapsulate it like this:

```python
from stream_infer.algo import BaseAlgo

class HeadDetectionAlgo(BaseAlgo):
    def init(self):
        self.model_id = 'damo/cv_tinynas_head-detection_damoyolo'
        self.head_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

    def run(self, frames):
        return self.head_detection(frames[0])
```

This way, you have completed the encapsulation and can call it normally in the future.

> [!CAUTION]
> In many cases, cuda or mps is used to accelerate inference. However, when using these accelerations **and needing real-time inference in the production environment after development**:
>
> The `run()` function inherited from `BaseAlgo` **must not return any tensors**! Try to manually convert to standard Python data formats, like dicts.
>
> Or copy the tensor (to CPU) for sharing between processes, as GPU tensors cannot be directly shared in a multi-process environment in real-time inference.

### Dispatcher

Dispatcher is a central service for playing and reasoning, used to cache inference frames, distribute inference frames, and collect inference time and result data.

Dispatcher provides functions for adding/getting frames and times. Stream Infer has a built-in [DevelopDispatcher](./stream_infer/dispatcher/develop.py) for manually storing and retrieving inference results.

You don't need to worry about the others, but to allow you to get the results and conveniently print or store them elsewhere, you should pay attention to the `collect()` function. Its source code implementation is as follows:

```python
def collect(self, position: int, algo_name: str, result):
    logger.debug(f"[{position}] collect {algo_name} result: {result}")
```

Based on this, if you wish to request the results to a REST service or perform other operations on existing data before the request, you can achieve this by **inheriting the Dispatcher class** and rewriting functions:

**Collect results to Redis**

```python
class RedisDispatcher(Dispatcher):
    def __init__(
        self, buffer: int, host: str = "localhost", port: int = 6379, db: int = 0
    ):
        super().__init__(buffer)
        self.conn = redis.Redis(host=host, port=port, db=db)

    def collect(self, position: int, algo_name: str, result):
        key = f"results:{algo_name}"
        self.conn.zadd(key, {result: position})
```

**Request results to REST**

```python
from stream_infer.dispatcher import Dispatcher
import requests
...
class RequestDispatcher(Dispatcher):
    def __init__(self, buffer):
        super().__init__(buffer)
        self.sess = requests.Session()
        ...

    def collect(self, position: int, algo_name: str, result):
        req_data = {
            "position": position
            "algo_name": algo_name
            "result": result
        }
        self.sess.post("http://xxx.com/result/", json=req_data)
```

Then instantiate:

```python
# Offline inference
dispatcher = RequestDispatcher.create(mode="offline", buffer=30)
# Real-time inference
dispatcher = RedisDispatcher.create(mode="realtime", buffer=15, host="localhost", port=6379, db=1)
```

You may have noticed that the instantiation of dispatcher differs between offline and real-time inference. This is because **in real-time inference, playback and inference are not in the same process**, and both need to share the same dispatcher, only the mode parameter has been changed, but the internal implementation uses the DispatcherManager agent.

> [!CAUTION]
> For the `buffer` parameter, the default value is 30, which keeps the latest 30 frames of ndarray data in the buffer. **The larger this parameter, the more memory the program occupies!**
>
> It is recommended to set it to `buffer = max(frame_count * (frame_step if frame_step else 1))` based on your algorithm requirements. For example, if you have an algorithm that needs `frame_count=5` and `frame_step=3`, you should set `buffer` to at least 15 to ensure enough frames are available.

### Player

Player inputs dispatcher, producer, and the video/stream address for playback and inference.

```python
from stream_infer import Player

...

player = Player(dispatcher, producer, source, show_progress)
```

The `show_progress` parameter defaults to True, in which case the tqdm is used to display the progress bar. When set to False, progress is printed through the logger.

### Inference

Inference is the core of the framework, implementing functions such as loading algorithms and running inference.

An Inference object requires a Dispatcher object and Player object for playback, frame retrieval and sending inference results.

```python
from stream_infer import Inference

...

inference = Inference(dispatcher, player)
```

When you need to load an algorithm, for example from the [BaseAlgo](#basealgo) section:

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(dispatcher, player)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

Here, we can give HeadDetectionAlgo a name to identify the running algorithm (needed when collecting in Dispatcher and to avoid duplicates).

The parameters for loading an algorithm are the framework's core functionality, allowing you to freely implement frame retrieval logic:

- **frame_count**: The number of frames the algorithm processes each time it runs. For example:

  - `frame_count=1`: Process only the most recent frame (for single-image algorithms like face detection)
  - `frame_count=5`: Process 5 frames simultaneously (for algorithms needing short temporal context like simple action recognition)
  - `frame_count=30`: Process 30 frames (about 1 second of video at 30fps) for algorithms requiring longer temporal context

- **frame_step**: The sampling interval between frames. Controls how frames are selected from the buffer:

  - `frame_step=0`: Get the most recent `frame_count` consecutive frames
  - `frame_step=1`: Get every frame (same as 0 but more explicit)
  - `frame_step=2`: Get every other frame (skip one frame between each selected frame)
  - `frame_step=10`: Get every 10th frame (useful for analyzing changes over longer time periods)

- **interval**: The time interval (in seconds) between algorithm executions. Controls how frequently the algorithm runs:
  - `interval=0.1`: Run algorithm 10 times per second (for high-frequency applications like tracking)
  - `interval=1.0`: Run algorithm once per second (for general real-time analysis)
  - `interval=5.0`: Run algorithm every 5 seconds (for slow-changing scenes or computationally intensive algorithms)

### Producer

Producer loads videos or streams using different methods, such as PyAV, OpenCV, etc., and adjusts or transforms the frames in terms of width, height, and color space, eventually returning each frame as a numpy array.

Instantiating a Producer often requires inputting the frame width, height, and color order required for inference. The default color order is the same as the BGR order returned by `cv2.imread()`.

```python
from stream_infer.producer import PyAVProducer, OpenCVProducer

producer = PyAVProducer(1920, 1080)
producer = OpenCVProducer(1920, 1080)
```

> [!NOTE]
> In most cases `OpenCVProducer` is sufficient and performs well. However, you may still need to use `PyAVProducer` (based on ffmpeg) to load some videos or streams that OpenCV cannot decode

### Run

Simply run the entire script through Inference's `start()`.

```python
inference.start(player, fps=fps, position=0, mode="offline", recording_path="./processed.mp4")
```

- fps: Indicates the desired playback frame rate. **If the frame rate of the video source is higher than this number, it will skip frames through frame skipping logic to forcibly play at this specified frame rate**, thereby saving performance to some extent.
- position: Accepts a parameter in seconds, which can specify the start position for inference (only available in offline inference; how could you specify a position in real-time inference, right?).
- mode: The default is `realtime`.
- recording_path: After this parameter is added, the processed frame can be recorded into a new video file under offline inference.

It should be specifically noted that during the inference process, you may need to process the frames or inference results. We provide a `process` decorator and function to facilitate this purpose.

> [!WARNING]
> In a real-time inference environment, due to the reason of running multiple processes, it is not possible to use a decorator to set up the process procedure

Currently, the recorded videos only support the mp4 format. When you use `OpenCVProducer`, the files are encoded in mp4v, while under `PyAVProducer`, they are encoded in h264 mp4 format. We recommend using `PyAVProducer` as it offers a better compression rate.

For specific usage, you can refer to the example codes in [examples/offline_custom_process_record.py](./examples/offline_custom_process_record.py) and [examples/realtime_custom_process.py](./examples/realtime_custom_process.py).

## License

Stream Infer is licensed under the [Apache License](LICENSE).
