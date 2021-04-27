import abc
import torch.nn as nn

from common.torch_utils import TorchUtils
from network.abstract_network import AbstractNetwork


class AbstractValueNetwork(AbstractNetwork, abc.ABC):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractNetwork.__init__(self, input_dim, output_dim, network_setting, device)

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def get_value(self, input):
        pass
