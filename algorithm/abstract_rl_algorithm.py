import abc


class AbstractRLAlgorithm(abc.ABC):
    def __init__(self, env):
        self.env = env
        self.policy_network = None
        self.value_network = None

        self.buffer = None

    def set_policy_network(self, policy_network):
        self.policy_network = policy_network

    def set_value_network(self, value_network):
        self.value_network = value_network

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def optimize_policy_network(self):
        pass

    @abc.abstractmethod
    def opimize_value_network(self):
        pass

    @abc.abstractmethod
    def estimate_policy_loss(self):
        pass

    @abc.abstractmethod
    def estimate_value_loss(self):
        pass