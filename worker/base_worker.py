from abc import ABC, abstractmethod


class BaseWorker(ABC):
    def __init__(self, policy_network, buffer, value_network):
        self.policy_network = policy_network
        self.value_network = value_network
        self.buffer = buffer

    @abstractmethod
    def sample_trajectory(self, sample_size=-1, rendering=False):
        pass
