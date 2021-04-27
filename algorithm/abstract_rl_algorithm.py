import abc


class AbstractRLAlgorithm(abc.ABC):
    def __init__(self, env, policy_network, value_network):
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network

        self.buffer = None

        self.policy_network_optimizer = None
        self.value_network_optimizer = None

    def set_policy_network(self, policy_network):
        self.policy_network = policy_network

    def set_value_network(self, value_network):
        self.value_network = value_network

    @abc.abstractmethod
    def train(self):
        pass

    def optimize_policy_network(self, loss):
        self.policy_network_optimizer.zero_grad()
        loss.backward()
        self.policy_network_optimizer.step()

    def optimize_value_network(self, loss):
        self.value_network_optimizer.zero_grad()
        loss.backward()
        self.value_network_optimizer.step()

    @abc.abstractmethod
    def estimate_policy_loss(self, *args):
        pass

    @abc.abstractmethod
    def estimate_value_loss(self, *args):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass