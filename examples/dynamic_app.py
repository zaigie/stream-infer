from stream_infer.dynamic import DynamicApp
from stream_infer.log import logger

config = {
    "mode": "offline",
    "source": "./classroom.mp4",
    "fps": 30,
    "dispatcher": {
        "module": "stream_infer.dispatcher",
        "name": "DevelopDispatcher",
        "args": {"buffer": 1},
    },
    "algos": [
        {
            "module": "./algos/yolo.py",
            "name": "YoloDetectionAlgo2",
            "args": {
                "frame_count": 1,
                "frame_step": 0,
                "interval": 1,
            },
        }
    ],
    "producer": {"type": "opencv", "width": 640, "height": 360},
}


def process(inference, *args, **kwargs):
    _, data = inference.dispatcher.get_last_result("YoloDetectionAlgo2", clear=True)
    if data is not None:
        logger.debug(f"{data}")


if __name__ == "__main__":
    app = DynamicApp(config)
    app.process(process)
    app.start()
