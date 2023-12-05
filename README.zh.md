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

以下是一个 Stream Infer 的简单示例，以帮助您开始使用：

```python
from stream_infer import Inference, FrameTracker, TrackerManager, Player
from stream_infer.algo import BaseAlgo
from stream_infer.exporter import BaseExporter
from stream_infer.producer import PyAVProducer
from stream_infer.log import logger

import time

class ExampleAlgo(BaseAlgo):
    def init(self):
        logger.info(f"{self.name} 已初始化")

    def run(self, frames):
        logger.debug(f"{self.name} 开始推理 {len(frames)} 帧")
        time.sleep(0.3)
        result = {"name": self.name}
        logger.debug(f"{self.name} 推理完成: {result}")
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

## 功能与概念

### 实时推理

![时序](./docs/img/real-time.png)

实时推理是指输入一个视频或流，视频或流以正常的现实时间播放速度进行播放，并添加帧到帧轨道中，播放进程与推理进程独立，由于推理无论如何都会花费一定时间，所以会产生或大或小的结果延迟，但不会造成内存泄漏和堆积。

### 离线推理

**处理性能好**

![](./docs/img/offline_good.png)

**处理性能差**

![](./docs/img/offline_bad.png)

离线推理是指输入一个视频（这里不能输入流了），以当前计算机能处理的速度，在取帧的同时串行执行推理，取帧与推理交错，由于推理无论如何都会花费一定时间，根据机器性能，整个进程的运行时间可能大于也可能小于视频时长。

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

### Step2. Exporter

Exporter 用来收集推理结果，可以轻松地收集算法名称、结果、时间等数据。

目前 Exporter 只实现了 BaseExporter，更多的如 RedisExporter、TDEngineExporter、MySQLExporter 等等将会很快开发好。

以下是 BaseExporter 的简单源码和一个名为 PrintExporter 的实现：

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

打印后，你会得到最后一个推理结果，`inference_result` 的结构是 `(current_time, algo_instance.name, result)`

### Step3. Producer

Producer 通过不同方式，如 PyAV、FFmpeg、ImageIO 等，加载视频或流媒体，并从对帧的宽高、色彩空间等进行调整或转化，最终将每一帧返回为 numpy 数组。

实例化一个 Producer 往往需要输入推理需要的帧宽高和色彩空间，色彩默认为 BGR24

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

一个 Inference 对象须要输入 FrameTracker 对象（取帧）和 Exporter 对象（收集）

```python
from stream_infer import Inference
inference = Inference(frame_tracker, exporter)
```

当你需要加载算法时，这里以 Step1 中的例子举例

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(frame_tracker, exporter)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

其中，我们可以为 HeadDetectionAlgo 指定一个 name，用于标识运行的算法名称（在 Exporter 收集的时候需要且避免重复），同时注意一下参数：

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

事实上，一般**不在**实际生产开发中执行 Quick Start 里最后运行推理的代码：

```python
# ！！！不要在实际生产开发中用
inference.start(player, fps=play_fps, is_offline=OFFLINE)
```

它将离线和实时两个运行方式做了合并，但这是不太正确的，因为我们往往需要在运行过程中 debug 或者处理其它任务，让我们看看 `start()` 函数的源码

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

从中可以很简单地抽出，实时运行和离线运行下的关键函数

#### 实时运行

```python
player.play_realtime(fps)
while player.is_active():
    print(player.get_current_time_str())
    self.run_inference()
    print(exporter.send())
    # Other action
```

#### 离线运行

```python
for _, current_frame in player.play(fps):
    self.auto_run_specific_inference(player.fps, current_frame)
    print(exporter.send())
    # Other action
```

## 许可证

Stream Infer 根据 [Apache 许可证](LICENSE) 授权。
