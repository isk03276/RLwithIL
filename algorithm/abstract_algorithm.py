from abc import ABC, abstractmethod


class AbstractAlgorithm(ABC):
    def __init__(self, env, policy_network):
        self.env = env
        self.policy_network = policy_network

        self.buffer = None

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
    def train(self):
        pass

    def optimize_policy_network(self, loss):
        self.policy_network_optimizer.zero_grad()
        loss.backward()
        self.policy_network_optimizer.step()

    @abstractmethod
    def estimate_policy_loss(self, *args):
        pass

    @abstractmethod
    def __str__(self):
        pass