import time

from .log import logger


class Timer:
    def __init__(self, interval, key=None):
        self.interval = interval
        self.last_call = None
        self.key = key

    def wait(self):
        current_time = time.time()

        if self.last_call is not None:
            elapsed = current_time - self.last_call
            if elapsed > self.interval:
                pass
            else:
                time.sleep(self.interval - elapsed)

        self.last_call = time.time()

    def is_time(self) -> bool:
        current_time = time.time()
        elapsed = round(current_time - self.last_call, 3) if self.last_call else 0
        if self.last_call is None or elapsed >= self.interval:
            if elapsed > self.interval:
                latency = round(elapsed - self.interval, 3)
                suggest = round(self.interval + latency, 1)
                logger.warning(
                    f"{self.key} is {latency}s later than the expected interval ({self.interval}s). \nPlease change the interval >= {suggest} or optimize your algorithm"
                )
            self.last_call = current_time
            return True
        return False
