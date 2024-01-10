import threading as th
from typing import Union, List

from .dispatcher import Dispatcher
from .algo import BaseAlgo
from .player import Player
from .recorder import Recorder
from .timer import Timer
from .model import Mode
from .log import logger


class Inference:
    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher
        self.inferences_info = []
        self.timers = {}

        self.is_stop = False
        self.process_func = self.default_process

    def load_algo(
        self,
        algo_instance,
        frame_count: int,
        frame_step: int,
        interval: Union[int, float],
    ):
        if not isinstance(algo_instance, BaseAlgo):
            err = f"Algo instance must be an instance of `BaseAlgo`, but got {type(algo_instance)}"
            logger.error(err)
            raise ValueError(err)
        self.inferences_info.append((algo_instance, frame_count, frame_step, interval))
        self.timers[algo_instance.name] = Timer(interval, key=algo_instance.name)
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

    def auto_run_specific(self, fps: int, current_frame_index: int) -> List[str]:
        current_algo_names = []
        for algo_instance, _, _, frequency in self.inferences_info:
            if current_frame_index % int(frequency * fps) == 0:
                self.run_specific(algo_instance.name)
                current_algo_names.append(algo_instance.name)
        return current_algo_names

    def run_specific(self, algo_name: str):
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
        self.dispatcher.collect(
            self.dispatcher.get_current_position(), algo_instance.name, result
        )

    def default_process(self, *args, **kwargs):
        pass

    def process(self, func):
        def wrapper(*args, **kwargs):
            return func(self, *args, **kwargs)

        self.process_func = wrapper

    def start(
        self,
        player: Player,
        fps: int = 30,
        position: int = 0,
        mode: Mode = Mode.REALTIME,
        recording_path: str = None,
    ):
        if mode in [Mode.OFFLINE, Mode.OFFLINE.value]:
            recorder = Recorder(player, recording_path) if recording_path else None
            for frame, current_frame_index in player.play(fps, position):
                current_algo_names = self.auto_run_specific(fps, current_frame_index)
                processed_frame = self.process_func(
                    frame=frame, current_algo_names=current_algo_names
                )
                frame = processed_frame if processed_frame is not None else frame
                if recorder:
                    recorder.add_frame(frame)
            if recorder:
                recorder.close()
        else:
            player.play_async(fps)
            self.run_async()
            while player.is_active():
                self.process_func()
            self.stop()
            player.stop()
        self.dispatcher.clear()
