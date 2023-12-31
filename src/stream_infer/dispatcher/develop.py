from .base import Dispatcher
from ..log import logger


class DevelopDispatcher(Dispatcher):
    """
    For develop
    """

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
