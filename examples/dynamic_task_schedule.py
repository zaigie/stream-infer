from stream_infer.dynamic import DynamicApp, DynamicConfig
import tasktiger

tiger = tasktiger.TaskTiger()


@tiger.task(queue="dynamic")
def dynamic_task(config: DynamicConfig):
    dynamic = DynamicApp(config)
    dynamic.start()


# dynamic_task.delay(config)
