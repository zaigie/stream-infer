# Stream Infer

<p align="left">
   <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

Stream Infer 是一个为视频处理应用中的流式推理设计的 Python 库。它包含用于视频帧生成、推理算法和结果导出的模块化组件。

## 安装

```bash
pip install stream-infer
```

## 快速开始

以下是一个 Stream Infer 的简单示例，以帮助您开始使用

该示例用了 [ModelScope](https://modelscope.cn/models/damo/cv_tinynas_head-detection_damoyolo/summary) 上的一个开源垂类检测模型，用于检测人头。

```python
from stream_infer import Inference, FrameTracker, TrackerManager, Player
from stream_infer.algo import BaseAlgo
from stream_infer.collector import BaseCollector
from stream_infer.producer import PyAVProducer, OpenCVProducer
from stream_infer.log import logger

import time
import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class HeadDetectionAlgo(BaseAlgo):
    def init(self):
        self.model_id = "damo/cv_tinynas_head-detection_damoyolo"
        self.head_detection = pipeline(
            Tasks.domain_specific_object_detection, model=self.model_id
        )

    def run(self, frames):
        logger.debug(f"{self.name} 开始推理 {len(frames)} 帧")
        result = self.head_detection(frames[0])
        logger.debug(f"{self.name} 推理完成: {result}")
        return result


class Collector(BaseCollector):
    def get(self, name):
        if self.results.get(name):
            return self.results[name]
        return None

    def get_last(self, name):
        algo_results = self.get(name)
        if algo_results is not None and len(algo_results.keys()) > 0:
            return algo_results[(str(max([int(k) for k in algo_results.keys()])))]
        return None


INFER_FRAME_WIDTH = 1920
INFER_FRAME_HEIGHT = 1080
OFFLINE = True

video = "/path/to/your/video.mp4"
fps = 30

# producer = PyAVProducer(INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT)
producer = OpenCVProducer(INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT)
collector = Collector()

if __name__ == "__main__":
    max_size = 300
    frame_tracker = (
        FrameTracker(max_size) if OFFLINE else TrackerManager().create(max_size)
    )

    inference = Inference(frame_tracker, collector)
    inference.load_algo(HeadDetectionAlgo(), frame_count=1, frame_step=fps, interval=1)

    player = Player(producer, frame_tracker, video)
    if OFFLINE:
        for frame, current_frame in player.play(fps):
            inference.auto_run_specific_inference(player.fps, current_frame)
            data = collector.get_last("HeadDetectionAlgo")
            if data is None:
                continue

            if data is not None:
                # 绘制检测结果
                for box, label in zip(data["boxes"], data["labels"]):
                    start_point = (int(box[0]), int(box[1]))
                    end_point = (int(box[2]), int(box[3]))
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.rectangle(frame, start_point, end_point, color, thickness)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (start_point[0], start_point[1] - 10)
                    font_scale = 0.5
                    font_color = (0, 255, 0)
                    line_type = 2
                    cv2.putText(frame, label, org, font, font_scale, font_color, line_type)
            cv2.namedWindow("推理", cv2.WINDOW_NORMAL)
            cv2.imshow(f"推理", frame)
            cv2.waitKey(1)
    else:
        player.play_realtime(fps)
        while player.is_active():
            inference.run_inference()

    cv2.destroyAllWindows()
```

## 功能与概念

### 实时推理

![时序](./docs/img/real-time.png)

实时推理是指输入一个视频或流，视频或流以正常的现实时间播放速度进行播放，并添加帧到帧轨道中，播放进程与推理进程独立，由于推理无论如何都会花费一定时间，造成或大或小的结果延迟，但不会制造内存泄漏和堆积。

实时推理更常应用于 RTMP/RTSP/HLS 等流媒体的分析：

- 各类直播场景
- 实时监控
- 实时会议
- 临床手术
- ...

### 离线推理

**处理性能好**

![](./docs/img/offline_good.png)

**处理性能差**

![](./docs/img/offline_bad.png)

离线推理是指输入一个视频（这里不能输入流了），以当前计算机能处理的速度，在取帧的同时串行执行推理，取帧与推理交错，由于推理无论如何都会花费一定时间，根据机器性能，整个进程的运行时间可能大于也可能小于视频时长。

离线推理应用于**所有非必须实时情况下**的视频结构化分析，如：

- 会后视频分析
- 手术视频复盘
- ...

**同时，由于离线推理的视频读取与算法推理是串行运行的，因此可以用于测试算法性能与效果（如快速开始中，通过 cv2 展示推理后的视频与算法数据），而实时推理则不适合算法开发阶段使用**

## 模块

![流程图](./docs/img/flow.svg)

### Step1. BaseAlgo

我们简单地将所有算法封装抽象为拥有 `init()` 和 `run()` 两个函数的类，这就是 BaseAlgo。

即使 Stream Infer 提供了关于流式推理的框架，但实际的算法仍然需要你自己编写，并在编写好后继承 BaseAlgo 类以统一封装调用。

比如，你已经完成了一个人头检测的算法，推理调用方式是：

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

那么，要在视频和流媒体中将该算法流式推理，请这样封装：

```python
from stream_infer.algo import BaseAlgo

class HeadDetectionAlgo(BaseAlgo):
    def init(self):
        self.model_id = 'damo/cv_tinynas_head-detection_damoyolo'
        self.head_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

    def run(self, frames):
        return self.head_detection(frames)
```

这样，你就完成了封装，并在以后能够正常调用它。

### Step2. Collector

Collector 用来收集推理结果，可以轻松地收集算法名称、结果、时间等数据。

目前 Collector 只实现了 BaseCollector，更多的如 RedisCollector、TDEngineCollector、MySQLCollector 等等将会很快开发好。

以下是 BaseCollector 的简单源码和一个名为 PrintCollector 的实现：

```python
class BaseCollector:
    def __init__(self):
        self.results = {}

    def collect(self, inference_result):
        if inference_result is not None:
            time = inference_result[0]
            name = inference_result[1]
            data = inference_result[2]
            self.results[name][time] = data

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def aggregate(self):
        raise NotImplementedError

    def clear(self):
        self.results.clear()


class PrintCollector(BaseCollector):
    def get(self, name):
        if self.results.get(name):
            return self.results[name]
        return None

    def get_all(self):
        return self.results
```

你可以通过 `collector.get(algo_instance.name)` 获取指定算法名称的结果，也可以通过 `collector.get_all()` 打印所有推理结果，在不重写 `collect()` 方法的情况下，`get_all()` 返回的格式大致如下：

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

### Step3. Producer

Producer 通过不同方式，如 PyAV、OpenCV、ImageIO（仅适用于离线） 等，加载视频或流媒体，并从对帧的宽高、色彩空间等进行调整或转化，最终将每一帧返回为 numpy 数组。

实例化一个 Producer 往往需要输入推理需要的帧宽高和色彩顺序，默认的色彩顺序与 `cv2.imread()` 返回的 BGR 顺序相同。

```python
from stream_infer.producer import PyAVProducer

producer = PyAVProducer(1920, 1080)
```

### Step4. FrameTracker

FrameTracker 可以理解为一个帧轨道，支持缓存最近的帧到队列中，可设置队列 `max_size`（默认为 120 帧）。

同时，FrameTracker 还兼任了根据 fps 和当前帧得到当前播放时间的职责。

需要注意的是，实时和离线是两种不同的运行模式，实时运行时，producer 和 inference 两者不在一个进程中，属于多进程环境，而 producer 和 inference 又都需要访问同一个 FrameTracker 对象，因此，当你需要实时运行流式推理时，需要通过 TrackerManager 创建一个 FrameTracker 对象，而不是直接通过 FrameTracker 实例化。

```python
from stream_infer import FrameTracker, TrackerManager

# 离线环境下
frame_tracker = FrameTracker()

# 实时环境下
frame_tracker = TrackerManager().create(max_size=150)
```

### Step5. Inference

Inference 是本框架的核心，加载算法、运行推理等功能都由它实现。

一个 Inference 对象须要输入 FrameTracker 对象（取帧）和 Collector 对象（收集）

```python
from stream_infer import Inference

inference = Inference(frame_tracker, collector)
```

当你需要加载算法时，这里以 Step1 中的例子举例

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(frame_tracker, collector)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

其中，我们可以为 HeadDetectionAlgo 指定一个 name，用于标识运行的算法名称（在 Collector 收集的时候需要且避免重复），同时注意一下参数：

- frame_count：算法需要获取的帧数量，也就是最终 run() 函数中收到的 frames 数量。
- frame_step：表示每隔 `frame_step` 取 1 帧，共取 `frame_count` 帧，如果该参数填入 fps，那么就意味着每秒取最后的 `frame_count` 帧。
- interval：算法调用频率，如上的 `AnyOtherAlgo` 就只会在一分钟才调用一次

### Step6. Player & run

Player 输入 producer、frame_tracker 和视频/流媒体地址进行播放与推理

一个 Player 须要如上参数

```python
from stream_infer import Player
player = Player(producer, frame_tracker, video)
```

加载好了算法、有了收集器和帧轨道以及播放器，就可以开始运行推理了

#### 实时运行

```python
player.play_realtime(fps)
while player.is_active():
    print(player.get_current_time_str())
    self.run_inference()
    print(collector.get_all())
    # Other action
```

#### 离线运行

```python
for _, current_frame in player.play(fps):
    current_algo_name = self.auto_run_specific_inference(player.fps, current_frame)
    if current_algo_name:
        print(collector.get(current_algo_name))
    # Other action
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
```

## 许可证

Stream Infer 根据 [Apache 许可证](LICENSE) 授权。
