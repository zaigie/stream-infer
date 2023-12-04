# Stream Infer

<p align="left">
   <strong>Englist</strong> | <a href="./README.zh.md">简体中文</a>
</p>

Stream Infer is a Python library designed for efficient streaming inference in video processing applications. It features modular components for video frame production, inference algorithms, and results exporting.

## Installation

```bash
pip install stream-infer
```

## Quick Start

Here's a simple example to get you started with Stream Infer:

````python
```python
from stream_infer import Inference, FrameTracker, TrackerManager, Player
from stream_infer.algo import BaseAlgo
from stream_infer.exporter import BaseExporter
from stream_infer.producer import PyAVProducer
from stream_infer.log import logger

import time

class ExampleAlgo(BaseAlgo):
    def init(self):
        logger.info(f"{self.name} init")

    def run(self, frames):
        logger.debug(f"{self.name} start infer {len(frames)} frames")
        time.sleep(0.3)
        result = {"name": self.name}
        logger.debug(f"{self.name} infer result: {result}")
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
````

## License

Stream Infer is licensed under the [Apache License](LICENSE).
