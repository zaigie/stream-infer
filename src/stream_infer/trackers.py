from collections import deque
from typing import List, Any


class FrameTracker:
    def __init__(self, max_size: int = 120):
        self.queue = deque(maxlen=max_size)
        self.current_time = 0

    def get_size(self) -> int:
        return len(self.queue)

    def add_frame(self, frame):
        self.queue.append(frame)

    def get_frames(self, count: int, step: int) -> List[Any]:
        if len(self.queue) < count * step:
            return []

        return list(self.queue)[-count * step :][::-1][::step][::-1]

    def increase_current_time(self):
        self.current_time += 1

    def get_current_time(self) -> int:
        return self.current_time
