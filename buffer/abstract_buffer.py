import abc


class AbstractBuffer(abc.ABC):
    def __init__(self, max_size):
        self.max_size = max_size
        self._buffer = []

    @abc.abstractmethod
    def add(self, *args):
        pass

    @abc.abstractmethod
    def sample(self, n):
        pass

    def clean_buffer(self):
        self._buffer = []


