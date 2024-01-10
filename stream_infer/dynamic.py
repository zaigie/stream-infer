import os
import sys
import importlib
from pydantic import BaseModel, ValidationError
from typing import Union

from .inference import Inference
from .player import Player
from .producer import OpenCVProducer, PyAVProducer
from .model import Mode, ProducerType
from .log import logger


class ProducerData(BaseModel):
    type: ProducerType
    width: int
    height: int


class DynamicImport(BaseModel):
    module: str
    name: str
    args: Union[dict, None] = None


class AlgoArgs(BaseModel):
    frame_count: int
    frame_step: int
    interval: int


class Algos(DynamicImport):
    args: AlgoArgs


class DynamicConfig(BaseModel):
    mode: Mode
    source: str
    fps: int
    dispatcher: DynamicImport
    algos: list[Algos]
    producer: ProducerData
    process: Union[DynamicImport, None] = None
    recording_path: Union[str, None] = None


class DynamicApp:
    def __init__(self, config: DynamicConfig) -> None:
        try:
            config = DynamicConfig(**config)
        except ValidationError as e:
            err = f"Invalid config: {e}"
            logger.error(err)
            raise e
        self.config = config
        dispatcher_module = self.dynamic_import(config.dispatcher.module)
        dispatcher_cls = getattr(dispatcher_module, config.dispatcher.name)
        self.dispatcher = dispatcher_cls.create(
            mode=config.mode, **config.dispatcher.args
        )
        self.inference = Inference(self.dispatcher)
        if config.process is not None:
            process_module = self.dynamic_import(config.process.module)
            self.inference.process(getattr(process_module, config.process.name))

    def start(self):
        if self.config.producer.type in [
            ProducerType.OPENCV,
            ProducerType.OPENCV.value,
        ]:
            producer = OpenCVProducer(
                self.config.producer.width, self.config.producer.height
            )
        elif self.config.producer.type in [ProducerType.PYAV, ProducerType.PYAV.value]:
            producer = PyAVProducer(
                self.config.producer.width, self.config.producer.height
            )
        else:
            raise ValueError(
                f"Unknown producer: {producer}, must be 'opencv' or 'pyav'"
            )
        for algo in self.config.algos:
            module = self.dynamic_import(algo.module)
            algo_class = getattr(module, algo.name)
            self.inference.load_algo(algo_class(), **algo.args.dict())
        self.inference.start(
            Player(
                self.dispatcher,
                producer,
                source=self.config.source,
                show_progress=False,
            ),
            fps=self.config.fps,
            mode=self.config.mode,
            recording_path=self.config.recording_path,
        )

    @staticmethod
    def dynamic_import(module_name):
        if os.path.isdir(module_name) or os.path.isfile(module_name):
            module_path = (
                os.path.dirname(module_name)
                if os.path.isfile(module_name)
                else module_name
            )
            module_name = os.path.basename(module_name).replace(".py", "")

            if module_path not in sys.path:
                sys.path.append(module_path)

        return importlib.import_module(module_name)
