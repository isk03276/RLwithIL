from buffer.abstract_buffer import AbstractBuffer


class OnPolicyBuffer(AbstractBuffer):
    def __init__(self, max_size):
        AbstractBuffer.__init__(self, max_size)

    def add(self, *, obs, acs, rews, nobs, dones):
        pass

    def sample(self, n):
        pass