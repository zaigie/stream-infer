import time


class Timer:
    def __init__(self, interval):
        self.interval = interval
        self.last_call = None

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
        if self.last_call is None or current_time - self.last_call >= self.interval:
            self.last_call = current_time
            return True
        return False
