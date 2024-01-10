from stream_infer import Inference
from stream_infer.log import logger


def dynamic_process(inference: Inference, *args, **kwargs):
    algos = inference.list_algos()
    for algo in algos:
        _, data = inference.dispatcher.get_last_result(algo, clear=True)
        if data is not None:
            logger.debug(f"{algo}: {data}")
