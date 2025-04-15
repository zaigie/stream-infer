from algos import YoloDetectionAlgo2

from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.log import logger
from stream_infer.producer import OpenCVProducer


def realtime_process(inference: Inference, *args, **kwargs):
    _, data = inference.dispatcher.get_last_result("YoloDetectionAlgo2", clear=True)
    if data is not None:
        logger.debug(f"{data}")


if __name__ == "__main__":
    dispatcher = DevelopDispatcher.create(logging_level="DEBUG")
    player = Player(
        dispatcher,
        OpenCVProducer(1920, 1080),
        source="./classroom.mp4",
        show_progress=False,
    )
    inference = Inference(dispatcher, player)

    inference.process(realtime_process)  # Set process func in real-time inference
    inference.load_algo(YoloDetectionAlgo2(), 1, 0, 1)
    inference.start()
