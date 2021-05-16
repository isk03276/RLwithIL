from network.abstract_value_network import AbstractValueNetwork
from common.torch_utils import TorchUtils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


class DiscreteMLPValueNetwork(AbstractValueNetwork):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractValueNetwork.__init__(self, input_dim, output_dim, network_setting, device)
        self.model = self._build_network()

    def forward(self, state):
        transformed_state = TorchUtils.transform_input(state)
        return self.model(transformed_state)

    def get_value(self, state):
        return self.forward(state)


class ContinuousMLPPolicyNetwork(AbstractValueNetwork):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractValueNetwork.__init__(self, input_dim, output_dim, network_setting, device)
        self.model = self._build_network()

        self.log_std = np.ones(self.output_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.Tensor(self.log_std))

    def forward(self, state):
        transformed_state = TorchUtils.transform_input(state)
        return self.model(transformed_state)

    def get_value(self, state):
        return self.forward(state)

class DiscreteCNNPolicyNetwork(AbstractValueNetwork):
    pass


class ContinuousCNNPolicyNetwork(AbstractValueNetwork):
    pass
