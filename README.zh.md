# Stream Infer

<p align="left">
   <a href="./README.md">Englist</a> | <strong>简体中文</strong>
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

## 许可证

Stream Infer 根据 [Apache 许可证](LICENSE) 授权。
