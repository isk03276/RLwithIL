import numpy as np
import random

from buffer.abstract_buffer import AbstractBuffer


class OnPolicyBuffer(AbstractBuffer):
    def __init__(self, max_size=-1):
        super().__init__(max_size)

    def add(self, ob, ac, ac_logprob, rew, nob, done, value, next_value):
        data = (ob, ac, ac_logprob, rew, nob, done, value, next_value)
        self._buffer.append(data)

    def sample(self, num_sample: int = -1, init_buffer: bool = True) -> np.ndarray:
        if num_sample == -1:
            result = np.array(self._buffer)
        else:
            result = np.array(random.sample(self._buffer, num_sample))

        if init_buffer:
            self._init_buffer()

        return (
            np.array(result[:, self.OB_IDX].tolist()),
            np.array(result[:, self.AC_IDX].tolist()),
            np.array(result[:, self.AC_LOGPROB_IDX].tolist()),
            np.array(result[:, self.REW_IDX].tolist()),
            np.array(result[:, self.NOB_IDX].tolist()),
            np.array(result[:, self.DONE_IDX].tolist()),
            np.array(result[:, self.VALUE_IDX].tolist()),
            np.array(result[:, self.NEXT_VALUE_IDX].tolist()),
        )

    def __len__(self):
        return len(self.buffer)
