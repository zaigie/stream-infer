from stream_infer.dynamic import DynamicApp

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
    # "process": {"module": "./dynamic_process.py", "name": "dynamic_process"},
    # "recording_path": "./processed.mp4",
}

# process and recording_path are optional

if __name__ == "__main__":
    app = DynamicApp(config)
    app.start()
