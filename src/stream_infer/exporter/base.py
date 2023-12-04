class BaseExporter:
    def __init__(self):
        self.results = []

    def collect(self, inference_result):
        if inference_result is not None:
            self.results.append(inference_result)

    def send(self, *args, **kwargs):
        raise NotImplementedError

    def send_all(self):
        for result in self.results:
            self.send(result)

    def clear(self):
        self.results.clear()
