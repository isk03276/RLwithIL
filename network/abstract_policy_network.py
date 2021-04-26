import abc


class AbstractPolicyNetwork(metaclass=abc.ABCMeta):
    def __init__(self, input_dim, output_dim, network_setting):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_setting = network_setting

    @abc.abstractmethod
    def build_network(self):
        pass

    @abc.abstractmethod
    def get_action(self, input):
        pass

    @abc.abstractmethod
    def get_actionprob(self, input):
        pass

    @abc.abstractmethod
    def get_entropy(self, input):
        pass
