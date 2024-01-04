from multiprocessing.managers import BaseManager
from collections import deque
from typing import List, Any

from ..log import logger


class Dispatcher:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
        self.current_time = 0
        self.current_frame = 0
        self.collect_results = {}

    def get_size(self) -> int:
        return len(self.queue)

    def add_frame(self, frame):
        self.queue.append(frame)
        self.current_frame += 1

    def get_frames(self, count: int, step: int) -> List[Any]:
        if step > 0:
            if len(self.queue) < count * step:
                return []

            return list(self.queue)[-count * step :][::-1][::step][::-1]
        elif step < 0:
            raise ValueError("step must be positive or zero")
        return list(self.queue)[-count:]

    def collect_result(self, inference_result):
        if inference_result is not None:
            time = str(inference_result[0])
            name = inference_result[1]
            data = inference_result[2]
            if self.collect_results.get(name) is None:
                self.collect_results[name] = {}
            self.collect_results[name][time] = data

    def clear_collect_results(self):
        self.collect_results.clear()

    def clear(self):
        self.queue.clear()
        self.collect_results.clear()
        self.current_time = 0
        self.current_frame = 0

    def increase_current_time(self):
        self.current_time += 1

    def get_current_time(self) -> int:
        return self.current_time

    def set_current_time(self, time):
        self.current_time = time

    def get_current_frame(self) -> int:
        return self.current_frame

    def set_current_frame(self, frame):
        self.current_frame = frame

    @classmethod
    def create(cls, max_size=30, offline=False):
        if offline:
            return cls(max_size)
        else:
            return DispatcherManager(cls).create(max_size)


class DispatcherManager:
    def __init__(self, obj=None):
        self._manager = None
        self._dispatcher = None
        self._obj = Dispatcher if obj is None else obj

    def create(self, max_size: int):
        if self._manager is None:
            self._initialize_manager(max_size)
        return self._dispatcher

    def _initialize_manager(self, max_size):
        BaseManager.register("Dispatcher", self._obj)
        self._manager = BaseManager()
        self._manager.start()
        self._dispatcher = self._manager.Dispatcher(max_size)
