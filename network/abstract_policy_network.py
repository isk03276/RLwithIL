import abc

from network.abstract_network import AbstractNetwork


class AbstractPolicyNetwork(AbstractNetwork, abc.ABC):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractNetwork.__init__(self, input_dim, output_dim, network_setting, device)

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def get_action(self, input):
        pass

    @abc.abstractmethod
    def get_ac_logprobs(self, input, output):
        pass

    @abc.abstractmethod
    def get_entropy(self, input):
        pass
