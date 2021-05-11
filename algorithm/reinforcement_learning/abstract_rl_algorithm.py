from abc import abstractmethod

from algorithm.abstract_algorithm import AbstractAlgorithm



class AbstractRLAlgorithm(AbstractAlgorithm):
    def __init__(self, env, policy_network, value_network):
        super().__init__(env, policy_network)

        self.value_network = value_network

        self.value_network_optimizer = None

    def set_value_network(self, value_network):
        self.value_network = value_network

    def optimize_value_network(self, loss):
        self.value_network_optimizer.zero_grad()
        loss.backward()
        self.value_network_optimizer.step()

    @abstractmethod
    def estimate_value_loss(self, *args):
        pass
