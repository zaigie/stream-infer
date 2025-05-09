# <img src="./docs/img/logo_.png?raw=true" alt="Stream Infer" height="60px">

[![PyPI](https://img.shields.io/pypi/v/stream-infer?color=dark-green)](https://pypi.org/project/stream-infer/)
[![PyPI downloads](https://img.shields.io/pypi/dm/stream-infer?color=dark-green)](https://pypi.org/project/stream-infer/)
[![GitHub license](https://img.shields.io/github/license/zaigie/stream-infer?color=orange)](./LICENSE)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/zaigie/stream-infer)](https://github.com/zaigie/stream-infer/graphs/commit-activity)

<p align="left">
   <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

Stream Infer 是一个为视频处理应用中的流式推理设计的 Python 库，可结合各种图像算法将视频结构化。支持实时和离线两种推理模式。

简而言之，Stream Infer 是一个不限制设备硬件与机器学习框架，支持云端或边缘 IoT 设备的轻量版 [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk)。

---

很多时候我们想要使用一个或多个图像算法和模型去分析视频，并能为这些算法设置不同的取帧逻辑和调用频率，最终得出结构化的推理结果。

有时候甚至需要接入一个实时的摄像头或者网络直播流，按照设定好的规则去推理并反馈结果。

如果你有以上需求，那么一个简单的 Stream Infer 就能满足你从开发到调试再到生产运行的一切需要。

![Flow](./docs/img/flow.cn.svg?raw=true)

## 功能特性

- [x] 依赖简洁，纯 Python 编写，不受硬件架构限制
- [x] 支持所有基于 Python 的算法部署框架
- [x] 不到 10 行代码即可运行一个视频推理任务
- [x] 支持离线推理和实时推理，只需要改一个参数
  - 离线推理将逐帧完全遍历视频并按照预设算法和逻辑串行推理得到结果
  - 实时推理将取帧与推理分开，依赖处理设备的性能产生或大或小的延迟
- [x] 离线推理支持录制视频文件到本地磁盘
- [x] 支持通过 [streamlit](https://github.com/streamlit/streamlit) 在本地/远程服务器上可视化开发和调试
- [x] 支持参数化动态调用，便于推理服务器开发
- [x] 模块低耦合，分工明确
- [ ] 实时推理下进行录制和推流

## 安装

```bash
pip install -U stream-infer
```

## 快速开始

所有示例依赖 YOLOv8 ，您可能需要额外通过 pip 工具安装其它包来使用这个示例：

```bash
pip install ultralytics
```

同时，示例中用到的视频文件在 [sample-videos](https://github.com/intel-iot-devkit/sample-videos)

### 离线推理

https://github.com/zaigie/stream_infer/assets/17232619/32aef0c9-89c7-4bc8-9dd6-25035bee2074

离线推理接收一个**有限长度的视频或流**，以当前计算机能处理的速度，在取帧的同时串行执行推理，取帧与推理交错。

由于推理无论如何都会花费一定时间，根据机器性能，整个进程的运行时间**可能大于也可能小于视频时长**。

除了用于开发阶段的调试外，离线推理在生产环境下也能应用于**所有非必须实时情况下**的视频结构化分析，如：

- 会后视频分析
- 手术视频复盘
- ...

查看及运行 demo：

- 常规运行：[examples/offline_general.py](./examples/offline_general.py)
- 设置处理函数处理帧和推理结果，并用 cv2 展示及录制：[examples/offline_custom_process_record.py](./examples/offline_custom_process_record.py)

> [!WARNING]
> 在 `offline_custom_process_record.py` 中使用了 OpenCV GUI 相关的功能，如展示窗口等，若要使用可以手动安装 opencv-python 或 opencv-contrib-python，亦或者：
>
> `pip install -U stream-infer[desktop]`

### 实时推理

实时推理接收一个**有限/无限长度的视频或流**，若有限则以正常的现实时间播放速度进行播放。

该模式下会固定维护一个可自定义大小的帧轨道，运行过程中会持续将当前帧添加到帧轨道中，**播放进程与推理进程独立**。

由于推理无论如何都会花费一定时间，且**播放不会等待推理结果**，所以必定会造成或大或小的推理结果与当前画面延迟。

实时推理不适合开发阶段用，常应用于 RTMP/RTSP/摄像头 等生产环境下实时场景的分析：

- 各类直播场景
- 实时监控
- 实时会议
- 临床手术
- ...

查看及运行 demo：

- 常规运行：[examples/realtime_general.py](./examples/realtime_general.py)
- 设置处理函数并手动打印推理结果：[examples/realtime_custom_process.py](./examples/realtime_custom_process.py)

### 动态执行

利用 Python 的反射与动态导入能力，支持通过 JSON 配置推理任务所需要的所有参数。

该模式主要可用于推理服务器的开发，通过 REST/gRPC 或其它方式传入结构化数据即可启动一个推理任务。

查看及运行 demo：[examples/dynamic_app.py](./examples/dynamic_app.py)

### 可视化开发&调试

https://github.com/zaigie/stream_infer/assets/17232619/6cbd6858-0292-4759-8d4c-ace154567f8e

通过 [streamlit](https://github.com/streamlit/streamlit) 实现的可视化 web 应用。

该模式主要用于本地/远程服务器上的算法开发和调试，支持自定义帧上绘制和数据展示组件。

要运行该功能，请安装 server 版本：

```bash
pip install -U 'stream-infer[server]'
```

查看及运行 demo：[examples/streamlit_app.py](./examples/streamlit_app.py)

```bash
streamlit run streamlit_app.py
```

## 模块

请结合 [examples](./examples) 阅读下述内容

### BaseAlgo

Stream Infer 简单地将所有算法封装抽象为拥有 `init()` 和 `run()` 两个函数的类，实现 BaseAlgo。

即使 Stream Infer 提供了关于流式推理的框架，但实际的算法功能仍然需要你自己编写，并在编写好后**继承 BaseAlgo 类**以统一封装调用。

比如，你已经完成了一个实时人头检测的算法，官方给予的推理调用方式是：

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

要在 Stream Infer 中正常使用请这样封装：

```python
from stream_infer.algo import BaseAlgo

class HeadDetectionAlgo(BaseAlgo):
    def init(self):
        self.model_id = 'damo/cv_tinynas_head-detection_damoyolo'
        self.head_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

    def run(self, frames):
        return self.head_detection(frames[0])
```

这样，你就完成了封装，并在以后能够正常调用它。

> [!CAUTION]
> 很多情况下会使用 cuda 或者 mps 对推理进行加速，但是请注意，当你使用上述两种加速**并需要在开发完成后的生产环境中实时推理**时：
>
> 继承 `BaseAlgo` 的 `run()` 函数中 **返回值不能有任何张量（tensor）** ！请尽量手动转化为标准的 Python 数据格式，如 dict 等。
>
> 或者将张量复制（到 CPU）再进行多进程之间的共享，这是因为实时推理在多进程环境下 GPU 张量不能直接共享。

### Dispatcher

Dispatcher 是播放和推理的中心服务，用来缓存推理帧、分发推理帧以及收集推理时间、结果数据。

Dispatcher 提供了帧以及时间的增加/获取函数。Stream Infer 内置了一个 [DevelopDispatcher](./stream_infer/dispatcher/develop.py) 用于手动存储并获取推理结果。

其它的您不用在意，但是为了让您能获取到结果并方便地打印、存储于其它位置，您需要关注 `collect()` 函数，它的源码实现如下：

```python
def collect(self, position: int, algo_name: str, result):
    logger.debug(f"[{position}] collect {algo_name} result: {result}")
```

在此基础上，若您想要将结果请求到 REST 服务，或者在请求前对现有数据做其它操作，都可以通过 **继承 Dispatcher 类** 并重写函数的方式实现。

**收集结果到 Redis**

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

**请求结果到 REST**

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

然后实例化：

```python
# 离线推理
dispatcher = RequestDispatcher.create(mode="offline", buffer=30)
# 实时推理
dispatcher = RedisDispatcher.create(mode="realtime", buffer=15, host="localhost", port=6379, db=1)
```

您可能注意到，在离线推理和实时推理下实例化 dispatcher 的方式不同，这是因为 **实时推理下播放与推理不在一个进程中** ，而两者都需要共享同一个 dispatcher，虽然只是改变了 mode 参数，但其内部实现使用了 DispatcherManager 代理。

> [!CAUTION]
> 对于 `buffer`参数，默认值为 30，会将最新的 30 帧 ndarray 数据存在缓冲区中，**该参数越大，程序占用的内存就越大！**
>
> 建议根据您的算法需求设置为 `buffer = max(frame_count * (frame_step if frame_step else 1))`。例如，如果您有一个算法需要 `frame_count=5` 和 `frame_step=3`，则应将 `buffer` 设置为至少 15，以确保有足够的帧可用。

### Player

Player 输入 dispatcher, producer 和视频/流媒体地址进行播放与推理

```python
from stream_infer import Player

...

player = Player(dispatcher, producer, source, show_progress)
```

`show_progress` 参数默认为 True，此时会使用 tqdm 显示进度条，而设置为 False 时会通过 logger 打印。

### Inference

Inference 是本框架的核心，加载算法、运行推理等功能都由它实现。

一个 Inference 对象须要输入 Dispatcher 对象和 Player 对象用以播放、取帧和发送推理结果等。

```python
from stream_infer import Inference

...

inference = Inference(dispatcher, player)
```

当你需要加载算法时，这里以 [BaseAlgo](#basealgo) 中的例子举例

```python
from anywhere_algo import HeadDetectionAlgo, AnyOtherAlgo

...

inference = Inference(dispatcher, player)
inference.load_algo(HeadDetectionAlgo("head"), frame_count=1, frame_step=fps, interval=1)
inference.load_algo(AnyOtherAlgo("other"), 5, 6, 60)
```

其中，我们可以为 HeadDetectionAlgo 指定一个 name，用于标识运行的算法名称（在 Dispatcher 收集的时候需要且避免重复）。

而加载算法的几个参数则是框架的核心功能，让您能自由实现取帧逻辑：

- **frame_count**：算法每次运行时处理的帧数量。例如：

  - `frame_count=1`：仅处理最新的一帧（适用于单帧图像处理算法，如人脸检测）
  - `frame_count=5`：同时处理 5 帧（适用于需要短时间上下文的算法，如简单动作识别）
  - `frame_count=30`：同时处理 30 帧（在 30fps 下约 1 秒视频），适用于需要更长时间上下文的算法

- **frame_step**：帧之间的采样间隔。控制如何从缓冲区中选择帧：

  - `frame_step=0`：获取最近的`frame_count`个连续帧
  - `frame_step=1`：获取每一帧（与 0 相同但更明确）
  - `frame_step=2`：每隔一帧获取一帧（每个选定帧之间跳过一帧）
  - `frame_step=10`：每隔 10 帧获取一帧（适用于分析较长时间范围内的变化）

- **interval**：算法执行之间的时间间隔（秒）。控制算法运行的频率：
  - `interval=0.1`：每秒运行算法 10 次（适用于高频应用如跟踪）
  - `interval=1.0`：每秒运行算法一次（适用于一般的实时分析）
  - `interval=5.0`：每 5 秒运行算法一次（适用于变化缓慢的场景或计算密集型算法）

### Producer

Producer 通过不同方式，如 PyAV、OpenCV 等，加载视频或流媒体，并从对帧的宽高、色彩空间等进行调整或转化，最终将每一帧返回为 numpy 数组。

实例化一个 Producer 往往需要输入推理需要的帧宽高和色彩顺序，默认的色彩顺序与 `cv2.imread()` 返回的 BGR 顺序相同。

```python
from stream_infer.producer import PyAVProducer, OpenCVProducer

producer = PyAVProducer(1920, 1080)
producer = OpenCVProducer(1920, 1080)
```

> [!NOTE]
> 大部分情况下 `OpenCVProducer` 够用，且性能优秀。不过您仍然可能需要使用 `PyAVProducer` （基于 ffmpeg）用来加载一些 OpenCV 无法解码的视频或流

### Run

通过 Inference 的 `start()` 即可简单运行整个脚本

```python
inference.start(player, fps=fps, position=0, mode="offline", recording_path="./processed.mp4")
```

- fps：表示期望播放帧率，**如果视频源的帧率大于这个数，将会由跳帧逻辑进行跳帧，强行播放这个指定的帧率**，一定程度上节省性能。
- position：接收一个以秒为单位的参数，可以指定开始推理的位置（仅离线推理下可用，实时推理怎么可能指定位置呢对吧？）。
- mode：默认为 `realtime`。
- recording_path：添加此参数后，在离线推理下可以录制处理后的帧到新的视频文件中。

需要特别提到的是，在推理过程中，您可能需要对帧或推理结果进行处理，我们提供了 `process` 装饰器和函数方便您完成这个目的。

> [!WARNING]
> 实时推理环境下，因为多进程运行的原因，不能使用装饰器设置 process 过程

目前，录制的视频只支持 mp4 格式，当您使用 `OpenCVProducer` 时，录制的是 mp4v 编码的文件，而在`PyAVProduer`下则是 h264 编码的 mp4 文件，我们更推荐您使用 `PyAVProducer`，因为它有更好的压缩率。

关于具体的使用您可以分别参考 [examples/offline_custom_process_record.py](./examples/offline_custom_process_record.py) 和 [examples/realtime_custom_process.py](./examples/realtime_custom_process.py) 的示例代码。

## 许可证

Stream Infer 根据 [Apache 许可证](LICENSE) 授权。
