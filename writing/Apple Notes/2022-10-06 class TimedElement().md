class TimedElement():
    def __init__(self, start_time="", end_time="", text=""):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
    def __str__(self) -> str:
        return f"\[{self.start_time},{self.end_time} {self.text}"
    def _is_valid_comparison(self, other):
        return (hasattr(other, "start_time") and
                hasattr(other, "end_time") and
                hasattr(other, "text"))
    def duration(self):
        return self.end_time - self.start_time
    def within(self, other):
        if not self._is_valid_comparison(other):
            return NotImplemented
        return (self.end_time <= other.end_time and
                self.start_time >= other.start_time)
    def __gt__(self, other):
        """Strictly greater than: self completely contains other"""
        return (self.start_time < other.start_time and self.end_time > other.end_time)
    def __lt__(self, other):
        """Strictly less than: other completely contains self"""
        return (self.start_time > other.start_time and self.end_time < other.end_time)
    def pct_overlap(self, other):
        if not hasattr(other, "duration"):
            return NotImplemented
        if self.duration() > other.duration():
            negative = True
            r = self
            l = other
        else:
            negative = False
            r = other
            l = self
        if l.start_time < r.start_time:
            duration = l.end_time - r.start_time
        else:
            duration = l.start_time - r.end_time
        duration = end - start
        pct = (duration / l.duration()) * 100
        if negative:
            return -pct
        else:
            return pct