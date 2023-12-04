from .timer import Timer


class Inference:
    def __init__(self, frame_queue, exporter=None):
        self.frame_queue = frame_queue
        self.exporter = exporter
        self.inferences_info = []
        self.timers = {}

    def load_algo(self, algo_instance, frame_count, frame_step, interval):
        self.inferences_info.append((algo_instance, frame_count, frame_step, interval))
        self.timers[algo_instance.name] = Timer(interval)
        algo_instance.init()

    def start(self, player, is_offline: bool = False):
        """
        Easy to use function to start inference with realtime mode.
        """
        if is_offline:
            for _, current_frame in player.run():
                self.auto_run_specific_inference(current_frame)
        else:
            player.play_realtime()
            while player.is_activate():
                self.run_inference()

    def run_inference(self):
        # start_time = time.time()

        for inference_info in self.inferences_info:
            algo_instance, _, _, _ = inference_info
            timer = self.timers[algo_instance.name]
            if timer.is_time():
                self._inference_task(inference_info)

        # end_time = time.time()
        # elapsed_time = end_time - start_time

    def auto_run_specific_inference(self, fps, current_frame):
        for algo_instance, _, _, frequency in self.inferences_info:
            if current_frame % int(frequency * fps) == 0:
                self.run_specific_inference(algo_instance.name)

    def run_specific_inference(self, algo_name):
        for inference_info in self.inferences_info:
            algo_instance, _, _, _ = inference_info
            if algo_instance.name == algo_name:
                self._inference_task(inference_info)

    def run_inference_loop(self):
        while True:
            self.run_inference()

    def _inference_task(self, inference_info):
        algo_instance, frame_count, frame_step, _ = inference_info
        frames = self.frame_queue.get_frames(frame_count, frame_step)
        if not frames:
            return -1
        result = algo_instance.run(frames)
        if self.exporter:
            self.exporter.collect(
                (self.frame_queue.current_time, algo_instance.name, result)
            )
        return result


class BaseAlgo:
    def __init__(self, name=None) -> None:
        if not name:
            name = self.__class__.__name__
        self.name = name

    def init(self):
        """
        Initialize the algo model or other resources.
        """
        raise NotImplementedError

    def run(self):
        """
        Run inference and return the inference result.
        """
        raise NotImplementedError
