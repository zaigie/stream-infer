from .base import Dispatcher
from ..log import logger


class DevelopDispatcher(Dispatcher):
    """
    For develop
    """

    def __init__(self, buffer: int):
        super().__init__(buffer)
        self.collect_results = {}

    def collect(self, position: int, algo_name: str, result):
        if self.collect_results.get(algo_name) is None:
            self.collect_results[algo_name] = {}
        self.collect_results[algo_name][str(position)] = result

    def get_result(self, name, clear=False):
        if self.collect_results.get(name):
            data = self.collect_results[name].copy()
            if clear:
                del self.collect_results[name]
            return data
        return None

    def get_last_result(self, name, clear=False):
        algo_results = self.get_result(name, clear)
        if algo_results is not None and len(algo_results.keys()) > 0:
            time_point = int(max([int(k) for k in algo_results.keys()]))
            return time_point, algo_results[str(time_point)]
        return -1, None

    def clear(self):
        super().clear()
        self.collect_results = {}
