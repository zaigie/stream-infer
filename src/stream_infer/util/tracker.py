import multiprocessing as mp
from multiprocessing.managers import BaseManager

from ..trackers import FrameTracker


def create_tracker_manager(max_size: int = 120):
    BaseManager.register("FrameTracker", FrameTracker)
    manager = BaseManager()
    manager.start()
    return manager.FrameTracker(max_size=max_size)
