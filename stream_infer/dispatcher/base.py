from multiprocessing.managers import BaseManager
from collections import deque
from typing import List, Any

from ..util import position2time
from ..log import logger
from ..model import Mode


class Dispatcher:
    def __init__(self, buffer: int):
        self.queue = deque(maxlen=buffer)
        self.current_position = 0
        self.current_frame_index = 0

    def add_frame(self, frame):
        self.queue.append(frame)
        self.current_frame_index += 1

    def get_frames(self, count: int, step: int) -> List[Any]:
        if step > 0:
            if len(self.queue) < count * step:
                return []

            return list(self.queue)[-count * step :][::-1][::step][::-1]
        elif step < 0:
            raise ValueError("step must be positive or zero")
        return list(self.queue)[-count:]

    def collect(self, position: int, algo_name: str, result):
        logger.debug(
            f"[{position2time(position)}] collect {algo_name} result: {result}"
        )

    def increase_current_position(self):
        self.current_position += 1

    def get_current_position(self) -> int:
        return self.current_position

    def set_current_position(self, time):
        self.current_position = time

    def get_current_frame_index(self) -> int:
        return self.current_frame_index

    def set_current_frame_index(self, frame):
        self.current_frame_index = frame

    def clear(self):
        self.queue.clear()
        self.current_position = 0
        self.current_frame_index = 0

    @classmethod
    def create(cls, mode: Mode = Mode.REALTIME, buffer: int = 30, **kwargs):
        if mode in [Mode.OFFLINE, Mode.OFFLINE.value]:
            return cls(buffer, **kwargs)
        else:
            return DispatcherManager(cls).create(buffer, **kwargs)


class DispatcherManager:
    def __init__(self, obj=None):
        self._manager = None
        self._dispatcher = None
        self._obj = Dispatcher if obj is None else obj

    def create(self, buffer: int, **kwargs):
        if self._manager is None:
            self._initialize_manager(buffer, **kwargs)
        return self._dispatcher

    def _initialize_manager(self, buffer, **kwargs):
        BaseManager.register("Dispatcher", self._obj)
        self._manager = BaseManager()
        self._manager.start()
        self._dispatcher = self._manager.Dispatcher(buffer, **kwargs)
