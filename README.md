# <img src="https://github.com/zaigie/stream-infer/blob/main/docs/img/logo_.png?raw=true" alt="Stream Infer" height="60px">

[![PyPI](https://img.shields.io/pypi/v/stream-infer?color=dark-green)](https://pypi.org/project/stream-infer/)
[![PyPI downloads](https://img.shields.io/pypi/dm/stream-infer?color=dark-green)](https://pypi.org/project/stream-infer/)
[![GitHub license](https://img.shields.io/github/license/zaigie/stream-infer?color=orange)](https://github.com/zaigie/stream-infer/blob/main/LICENSE)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/zaigie/stream-infer)](https://github.com/zaigie/stream-infer/graphs/commit-activity)

<p align="left">
   <strong>English</strong> | <a href="./README.zh.md">简体中文</a>
</p>

Stream Infer is a Python library designed for streaming inference in video processing applications, enabling the integration of various image algorithms for video structuring. It supports both real-time and offline inference modes.

In short, Stream Infer is a device hardware and ML framework agnostic, supporting a lightweight version of [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) for cloud or edge IoT devices.

---

Often we want to use one or more image algorithms and models to analyze videos, and be able to set different frame capture logics and invocation frequencies for these algorithms, ultimately obtaining structured inferential results.

Sometimes it's even necessary to connect to a real-time camera or a live web stream, to infer and feedback results according to pre-set rules.

If you have the above requirements, then a simple Stream Infer can meet all your needs from development to debugging and to production operation.

![Flow](https://github.com/zaigie/stream-infer/blob/main/docs/img/flow.svg?raw=true)

## Features

- Minimal dependencies, purely written in Python, not limited by hardware architecture
- Supports all Python-based algorithm deployment frameworks
- Run a video inference task in less than 10 lines code
- Supports both offline and real-time inference, just by changing one parameter
  - Offline inference completely traverses the video frame by frame and follows the preset algorithm and logical serial inference to get the result
  - Real-time inference separates frame fetching from inference and generates large or small delays depending on the performance of the processing device
- Offline inference supports recording video files to a local disk
- Supports visual development and debugging on local/remote servers via [streamlit](https://github.com/streamlit/streamlit)
- Modules are low-coupled, with clear division of labor

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

View and run the demo here: [examples/offline.py](https://github.com/zaigie/stream_infer/blob/main/examples/offline.py)

> [!WARNING]
> This example use OpenCV GUI-related features, such as presentation Windows, which can be manually installed either `opencv-python` or `opencv-contrib-python`, or simply
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

View and run the demo here: [examples/realtime.py](https://github.com/zaigie/stream_infer/blob/main/examples/realtime.py)

### Visualization Development & Debugging

https://github.com/zaigie/stream_infer/assets/17232619/6cbd6858-0292-4759-8d4c-ace154567f8e

Implemented through a visual web application using [streamlit](https://github.com/streamlit/streamlit).

> The current interface text is Chinese.

This mode is primarily used for algorithm development and debugging on local/remote servers, supporting custom frame drawing and data display components.

To run this feature, install the server version:

```bash
pip install -U 'stream-infer[server]'
```

View and run the demo here: [examples/server.py](https://github.com/zaigie/stream_infer/blob/main/examples/server.py)

```bash
streamlit run server.py
```

## Modules

Please read the following content in conjunction with [examples](https://github.com/zaigie/stream_infer/blob/main/examples).

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
        return self.head_detection(frames)
```

This way, you have completed the encapsulation and can call it normally in the future.

> [!CAUTION]
> In many cases, cuda or mps is used to accelerate inference. However, when using these accelerations **and needing real-time inference in the production environment after development**:
>
> The `run()` function inherited from `BaseAlgo` **must not return any tensors**! Try to manually convert to standard Python data formats, like dicts.
>
> Or copy the tensor (to CPU) for sharing between processes, as GPU tensors cannot be directly shared in a multi-process environment in real-time inference.

### Dispatcher

Dispatcher is the central service for playing and inferring, used to cache inference frames, distribute inference frames, and collect inference time and result data.

Dispatcher provides functions for adding/getting frames, adding/getting inference results and times.

You don't need to worry about other aspects, but to enable you to retrieve results and conveniently print or store them elsewhere, you should pay attention to the `collect_result()` function, implemented as follows:

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

Here, `inference_result` is the original result returned from inference, and the final `collect_results` format is roughly as follows:

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

Based on this, if you wish to request the results to a REST service or perform other operations on existing data before the request, you can achieve this by **inheriting the Dispatcher class** and rewriting functions:

```python
from stream_infer.dispatcher import Dispatcher
import requests
...
class RequestDispatcher(Dispatcher):
    def __init__(self, max_size: int = 120):
        super().__init__(max_size)
        self.sess = requests.Session()
        ...

    def collect_result(self, inference_result):
        super().__init__(inference_result)
        req_data = {
            "time": inference_result[0]
            "name": inference_result[1]
            "data": inference_result[2]
        }
        self.sess.post("http://xxx.com/result/", json=req_data)
...

# Offline inference
dispatcher = RequestDispatcher.create(offline=True, max_size=140)

# Real-time inference
dispatcher = RequestDispatcher.create(max_size=150)
```

> [!CAUTION]
> You may have noticed that the instantiation of dispatcher differs between offline and real-time inference. This is because **in real-time inference, playback and inference are not in the same process**, and both need to share the same dispatcher, only the offline parameter has been changed, but the internal implementation uses the DispatcherManager agent.

### Inference

Inference is the core of the framework, implementing functions such as loading algorithms and running inference.

An Inference object requires a Dispatcher object for frame retrieval and sending inference results.

```python
from stream_infer import Inference

...

inference = Inference(dispatcher)
```

When you need to load an algorithm, for example from the [BaseAlgo](#basealgo) section:

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(dispatcher)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

Here, we can give HeadDetectionAlgo a name to identify the running algorithm (needed when collecting in Dispatcher and to avoid duplicates).

The parameters for loading an algorithm are the framework's core functionality, allowing you to freely implement frame retrieval logic:

- frame_count: The number of frames the algorithm needs to get, which is the number of frames the run() function will receive.
- frame_step: Take 1 frame every `frame_step`, up to `frame_count` frames. (when `frame_count` is equal to 1, this parameter determines only the startup delay)
- interval: In seconds, indicating the frequency of algorithm calls, like `AnyOtherAlgo` will only be called once a minute to save resources when not needed.

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

### Player

Player inputs dispatcher, producer, and the video/stream address for playback and inference.

```python
from stream_infer import Player

...

player = Player(dispatcher, producer, video_path)
```

### Run

Simply run the entire script through Inference's `start()`.

```python
inference.start(player, fps=fps, position=0, offline=True, recording_path="./processed.mp4")
```

- fps: Indicates the desired playback frame rate. **If the frame rate of the video source is higher than this number, it will skip frames through frame skipping logic to forcibly play at this specified frame rate**, thereby saving performance to some extent.
- position: Accepts a parameter in seconds, which can specify the start position for inference (only available in offline inference; how could you specify a position in real-time inference, right?).
- offline: The default is False, whether it is offline inference, when you want to run real-time inference, no parameters can be passed.
- recording_path: After this parameter is added, the processed frame can be recorded into a new video file under offline inference.

It is worth mentioning that during the inference process, you may need to process the frames or inference results. We provide the `set_custom_process()` function to facilitate this purpose.

Currently, the recorded videos only support the mp4 format. When you use `OpenCVProducer`, the files are encoded in mp4v, while under `PyAVProducer`, they are encoded in h264 mp4 format. We recommend using `PyAVProducer` as it offers a better compression rate.

For specific usage, you can refer to the example codes in [examples/offline.py](https://github.com/zaigie/stream_infer/blob/main/examples/offline.py) and [examples/realtime.py](https://github.com/zaigie/stream_infer/blob/main/examples/realtime.py).

## License

Stream Infer is licensed under the [Apache License](LICENSE).
