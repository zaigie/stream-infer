class BaseCollector:
    def __init__(self):
        self.results = {}

    def collect(self, inference_result):
        if inference_result is not None:
            time = str(inference_result[0])
            name = inference_result[1]
            data = inference_result[2]
            if self.results.get(name) is None:
                self.results[name] = {}
            self.results[name][time] = data

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def aggregate(self):
        raise NotImplementedError

    def clear(self):
        self.results.clear()
