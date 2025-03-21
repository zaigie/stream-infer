import tasktiger

from stream_infer.dynamic import DynamicApp, DynamicConfig

tiger = tasktiger.TaskTiger()


@tiger.task(queue="dynamic")
def dynamic_task(config: DynamicConfig):
    dynamic = DynamicApp(config)
    dynamic.start()


# dynamic_task.delay(config)
