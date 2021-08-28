from abc import abstractmethod

from algorithm.base_algorithm import BaseAlgorithm


class BaseRLAlgorithm(BaseAlgorithm):
    def __init__(self, env, policy_network, value_network, lr):
        super().__init__(env, policy_network, lr)

        self.value_network = value_network
        self.value_network_optimizer = None

        self.worker = None

    def set_value_network(self, value_network):
        self.value_network = value_network

    def set_worker(self, worker):
        self.worker = worker

    @abstractmethod
    def estimate_value_loss(self, *args):
        pass
