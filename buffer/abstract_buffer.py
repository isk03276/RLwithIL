import abc


class AbstractBuffer(abc.ABC):
    OB_IDX = 0
    AC_IDX = 1
    AC_LOGPROB_IDX = 2
    REW_IDX = 3
    NOB_IDX = 4
    DONE_IDX = 5
    VALUE_IDX = 6
    NEXT_VALUE_IDX = 7

    def __init__(self, capacity):
        self.capacity = capacity
        self._init_buffer()

    def _init_buffer(self):
        self._buffer = []

    @abc.abstractmethod
    def add(self, *args):
        pass

    @abc.abstractmethod
    def sample(self, num_sample):
        pass

    def clean_buffer(self):
        self._buffer = []
