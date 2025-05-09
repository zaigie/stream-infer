from algos import YoloDetectionAlgo2

from stream_infer import Inference, Player
from stream_infer.dispatcher import Dispatcher
from stream_infer.producer import OpenCVProducer

source = "./classroom.mp4"
# source = "rtmp://"
# source = 0

if __name__ == "__main__":
    dispatcher = Dispatcher.create(logging_level="DEBUG")
    player = Player(
        dispatcher, OpenCVProducer(1920, 1080), source=source, show_progress=False
    )
    inference = Inference(dispatcher, player)

    inference.load_algo(YoloDetectionAlgo2(), frame_count=1, frame_step=0, interval=1)
    inference.start()
