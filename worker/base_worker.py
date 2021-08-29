from abc import ABC, abstractmethod

from network.abstract_policy_network import AbstractPolicyNetwork
from network.abstract_value_network import AbstractValueNetwork
from buffer.abstract_buffer import AbstractBuffer


class BaseWorker(ABC):
    def __init__(
        self,
        policy_network: AbstractPolicyNetwork,
        buffer: AbstractBuffer,
        value_network: AbstractValueNetwork,
    ):
        self.policy_network = policy_network
        self.value_network = value_network
        self.buffer = buffer

    def sample_trajectory(self, sample_size: int = -1, rendering: bool = False):
        data = next(self._sample_trajectory(sample_size, rendering))
        return data

    @abstractmethod
    def _sample_trajectory(self, sample_size: int = -1, rendering: bool = False):
        pass
