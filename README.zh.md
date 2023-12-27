# Stream Infer

<p align="left">
   <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

Stream Infer 是一个为视频处理应用中的流式推理设计的 Python 库。它包含用于视频帧生成、推理算法和结果导出的模块化组件。

## 安装

```bash
pip install -U stream-infer
```

## 快速开始

以下是一个 Stream Infer 的简单示例，以帮助您直接开始使用并了解 Stream Infer 做了什么工作

该示例用了 YOLOv8 的姿态模型进行检测并绘制结果到 cv2 窗口中

> 由于 Python 多线程的问题，实时推理下暂时没法展示到窗口

https://github.com/zaigie/stream_infer/assets/17232619/32aef0c9-89c7-4bc8-9dd6-25035bee2074

视频文件在 [sample-videos](https://github.com/intel-iot-devkit/sample-videos)

> 您可能需要额外通过 pip 工具安装其它包来使用这个示例：
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
            current_algo_name = inference.auto_run_specific(
                player.play_fps, current_frame
            )
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

## 功能与概念

### 实时推理

![时序](./docs/img/real-time.png)

实时推理是指输入一个视频或流，视频或流以正常的现实时间播放速度进行播放，并添加帧到帧轨道中，播放进程与推理进程独立，由于推理无论如何都会花费一定时间，造成或大或小的结果延迟，但设置好合理的播放帧率后，并不会制造内存泄漏和堆积。

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

![](./docs/img/offline_poor.png)

离线推理是指输入一个视频（这里不能输入流了），以当前计算机能处理的速度，在取帧的同时串行执行推理，取帧与推理交错，由于推理无论如何都会花费一定时间，根据机器性能，整个进程的运行时间可能大于也可能小于视频时长。

离线推理应用于**所有非必须实时情况下**的视频结构化分析，如：

- 会后视频分析
- 手术视频复盘
- ...

**同时，由于离线推理的视频读取与算法推理是串行运行的，因此可以用于测试算法性能与效果（如[快速开始](#快速开始)中，通过 cv2 展示推理后的视频与算法数据），而实时推理则不适合算法开发阶段使用**

## 模块

![流程图](./docs/img/flow.svg)

### BaseAlgo

我们简单地将所有算法封装抽象为拥有 `init()` 和 `run()` 两个函数的类，这就是 BaseAlgo。

即使 Stream Infer 提供了关于流式推理的框架，但**实际的算法功能仍然需要你自己编写**，并在编写好后继承 BaseAlgo 类以统一封装调用。

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

> [!CAUTION]
> 很多情况下，我们会使用 cuda 或者 mps 对推理进行加速，但是请注意，当你使用上述两种加速时：
>
> 在继承 `BaseAlgo` 的 `run()` 函数中**返回值不能有任何张量**！请尽量手动转化为标准的 Python 数据格式，如 dict 等。
>
> 或者将张量复制（到 CPU）再进行多进程之间的共享。
>
> 这是因为多进程环境中造成的，也可能是我学习地不够深入，如果有更好的解决办法我会尝试去解决。

### Dispatcher

Dispatcher 作为链接播放和推理的中心服务，用来缓存推理帧、分发推理帧以及收集推理时间、结果数据。

Dispatcher 提供了帧的增加/获取函数、推理结果以及时间的增加/获取函数，其它的您不用在意，但是为了让您能获取到结果并方便地打印、存储于其它位置，您需要关注 `collect_result()` 函数。

它们的源码实现如下：

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

其中 `inference_result` 是推理返回的结果，最终收集到的 `collect_results` 格式大致如下：

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

在此基础上，若您想要将结果请求到 REST 服务，或者在请求前对现有数据做其它操作，都可以通过**继承 Dispatcher 类**并重写函数的方式实现：

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

# 离线环境下
dispatcher = SelfDispatcher()

# 实时环境下
dispatcher = DispatcherManager(SelfDispatcher).create(max_size=150)
```

> 您可能注意到，在离线环境和实时环境下实例化 dispatcher 的方式不同，这是因为**实时环境下播放与推理不在一个进程中**，而两者都需要共享同一个 dispatcher，因此使用了 DispatcherManager 代理。

### Producer

Producer 通过不同方式，如 PyAV、OpenCV、ImageIO（仅适用于离线） 等，加载视频或流媒体，并从对帧的宽高、色彩空间等进行调整或转化，最终将每一帧返回为 numpy 数组。

实例化一个 Producer 往往需要输入推理需要的帧宽高和色彩顺序，默认的色彩顺序与 `cv2.imread()` 返回的 BGR 顺序相同。

```python
from stream_infer.producer import PyAVProducer, OpenCVProducer

producer = PyAVProducer(1920, 1080)
producer = OpenCVProducer(1920, 1080)
```

### Inference

Inference 是本框架的核心，加载算法、运行推理等功能都由它实现。

一个 Inference 对象须要输入 Dispatcher 对象用以取帧和发送推理结果等。

```python
from stream_infer import Inference

inference = Inference(dispatcher)
```

当你需要加载算法时，这里以 [BaseAlgo](#basealgo) 中的例子举例

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(dispatcher)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

其中，我们可以为 HeadDetectionAlgo 指定一个 name，用于标识运行的算法名称（在 Dispatcher 收集的时候需要且避免重复）。

而加载算法的几个参数则是框架的核心功能，让您能自由实现取帧逻辑：

- frame_count：算法需要获取的帧数量，也就是最终 run() 函数中收到的 frames 数量。
- frame_step：每隔 `frame_step` 取 1 帧，共取 `frame_count` 帧，如果该参数填入 fps，那么就意味着每秒取最后的 `frame_count` 帧。
- interval：单位秒，表示算法调用频率，如 `AnyOtherAlgo` 就只会在一分钟才调用一次，用来在不需要调用它的时候节省资源

### Player

Player 输入 dispatcher, producer 和视频/流媒体地址进行播放与推理

```python
from stream_infer import Player

player = Player(dispatcher, producer, video_path)
```

Player 有两个函数分别在实时与离线推理模式下执行

```python
player.play(fps=None, position=0)
player.play_async(fps=None)
```

两个函数都可以输入一个 fps 参数，这里表示的是播放时的帧率，**如果视频源的帧率大于这个数，将会由跳帧逻辑进行跳帧，强行播放这个指定的帧率**。这样也能一定程度上节省性能。

而在离线环境下，你还可以指定播放的位置，position 参数将接收一个以秒为单位的参数。

### Play & Run

#### 离线运行

Player 的 `play()` 返回一个可迭代对象，在循环中调用 `inference.auto_run_specific()` 即可根据当前帧索引自动推断应该运行哪个算法：

```python
if __name__ == "__main__":
    ...
    for frame, current_frame in player.play(PLAY_FPS):
        current_algo_name = inference.auto_run_specific(
            player.play_fps, current_frame
        )
        # 其它操作，比如绘制画面窗口
        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
        cv2.imshow("Inference", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
```

正如[离线推理](#离线推理)中描述的一样，上述所有的执行都是在一个进程、一个线程中同步进行的，因此你可以慢慢完成您想要的操作，比如算法效果检验（如[快速开始](#快速开始)中给出的获取推理结果并展示盒子到窗口等），即使因为同步运行会卡顿，但一切都是准确无误的。

#### 实时运行

Player 的 `play_async()` 和 Inference 的 `run_async()` 只需要运行即可：

> 需要特别注意的是，我们建议在实时运行的时候，播放帧率最好不要大于 30 帧，首先是因为过大的帧率对分析结果的准确度没有多少帮助，其次也是因为这样会导致内存泄漏帧堆积。

```python
if __name__ == "__main__":
    ...
    player.play_async(PLAY_FPS)
    inference.run_async()
    while player.is_active():
        pass
        # 其它操作
    inference.stop()
    player.stop()
```

通过 `player.is_active()` 监控播放状态，并在播放完后手动结束推理线程和播放进程。

## 许可证

Stream Infer 根据 [Apache 许可证](LICENSE) 授权。
