from algos import YoloDetectionAlgo

from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.producer import OpenCVProducer

dispatcher = DevelopDispatcher.create(mode="offline", buffer=1)
inference = Inference(dispatcher)

inference.load_algo(YoloDetectionAlgo(), frame_count=1, frame_step=0, interval=0.1)
inference.start(
    Player(dispatcher, OpenCVProducer(1920, 1080), source="./classroom.mp4")
)
