import time


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.perf_counter() - self.elapsed
            self.running = True

    @staticmethod
    def start_new():
        stopwatch = Stopwatch()
        stopwatch.start()
        return stopwatch

    def stop(self):
        if self.running:
            if self.start_time is None:
                raise ValueError("Stopwatch is running but start time is not set")
            self.elapsed = time.perf_counter() - self.start_time
            self.running = False

    def reset(self):
        self.start_time = None
        self.elapsed = 0
        self.running = False

    def elapsed_time(self):
        if self.running:
            if self.start_time is None:
                raise ValueError("Stopwatch is running but start time is not set")
            return time.perf_counter() - self.start_time
        return self.elapsed
