import abc
import torch.nn as nn


class AbstractPolicyNetwork(nn.Module, abc.ABC):
    def __init__(self, input_dim, output_dim, network_setting, device):
        super(AbstractPolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = network_setting.hidden_units
        self.hidden_activations = network_setting.hidden_activations
        self.output_activation = network_setting.output_activation
        self.num_layers = len(self.hidden_units)
        self.device = device

    @abc.abstractmethod
    def _build_network(self):
        pass

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def get_action(self, input):
        pass

    @abc.abstractmethod
    def get_entropy(self, input):
        pass
