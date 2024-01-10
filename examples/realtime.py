from stream_infer import Inference, Player
from stream_infer.dispatcher import Dispatcher
from stream_infer.producer import OpenCVProducer

from .algos import YoloDetectionAlgo2


def realtime_process(inference: Inference):
    pass


if __name__ == "__main__":
    dispatcher = Dispatcher.create()
    inference = Inference(dispatcher)
    inference.process(
        realtime_process
    )  # Set a process function when real-time inference
    inference.load_algo(YoloDetectionAlgo2(), frame_count=1, frame_step=0, interval=1)

    player = Player(
        dispatcher,
        OpenCVProducer(1920, 1080),
        source="./classroom.mp4",
        show_progress=False,
    )
    inference.start(player, fps=30)
