class LinearSchedule:
    def __init__(self, start: float, end: float = 0.0):
        self.start = start
        self.end = end

    def __call__(self, progress_remaining: float):
        return (self.start - self.end) * progress_remaining + self.end
