from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    def __init__(self, env, policy_network, lr):
        self.env = env
        self.lr = lr

        self.policy_network = policy_network
        self.policy_network_optimizer = None

    def set_policy_network(self, policy_network):
        self.policy_network = policy_network

    def get_policy_network(self):
        return self.policy_network

    def set_policy_network_optimizer(self, optimizer):
        self.policy_network_optimizer = optimizer

    def get_policy_network_optimizer(self):
        return self.policy_network_optimizer

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def estimate_policy_loss(self, *args):
        pass

    @abstractmethod
    def __str__(self):
        pass
