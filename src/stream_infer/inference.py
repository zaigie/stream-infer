import threading as th

from .timer import Timer
from .log import logger


class Inference:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.inferences_info = []
        self.timers = {}
        self.is_stop = False

    def load_algo(self, algo_instance, frame_count, frame_step, interval):
        self.inferences_info.append((algo_instance, frame_count, frame_step, interval))
        self.timers[algo_instance.name] = Timer(interval)
        algo_instance.init()

    def list_algos(self):
        result = []
        for algo_instance, _, _, _ in self.inferences_info:
            result.append(algo_instance.name)
        return result

    def run(self):
        for inference_info in self.inferences_info:
            algo_instance, _, _, _ = inference_info
            timer = self.timers[algo_instance.name]
            if timer.is_time():
                self._infer(inference_info)

    def run_loop(self):
        while not self.is_stop:
            self.run()

    def run_async(self):
        thread = th.Thread(target=self.run_loop)
        thread.start()
        return thread

    def stop(self):
        self.is_stop = True

    def auto_run_specific(self, fps, current_frame) -> str:
        current_algo_names = []
        for algo_instance, _, _, frequency in self.inferences_info:
            if current_frame % int(frequency * fps) == 0:
                self.run_specific(algo_instance.name)
                current_algo_names.append(algo_instance.name)
        return current_algo_names

    def run_specific(self, algo_name):
        for inference_info in self.inferences_info:
            algo_instance, _, _, _ = inference_info
            if algo_instance.name == algo_name:
                self._infer(inference_info)

    def _infer(self, inference_info):
        algo_instance, frame_count, frame_step, _ = inference_info
        frames = self.dispatcher.get_frames(frame_count, frame_step)
        if not frames:
            return -1
        result = algo_instance.run(frames)
        self.dispatcher.collect_result(
            (self.dispatcher.get_current_time(), algo_instance.name, result)
        )
        return result
