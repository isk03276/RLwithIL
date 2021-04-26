from network.abstract_policy_network import AbstractPolicyNetwork
from common.torch_utils import TorchUtils

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np


class DiscreteMLPPolicyNetwork(AbstractPolicyNetwork):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractPolicyNetwork.__init__(self, input_dim, output_dim, network_setting, device)
        self.model = self._build_network()

    def _build_network(self):
        model = []
        last_input_dim = self.input_dim
        for i in range(self.num_layers):
            model.append(nn.Linear(last_input_dim, self.hidden_units[i]))
            model.append(TorchUtils.get_activation_fn(self.hidden_activations[i])())
            last_input_dim = self.hidden_units[i]
        model.append(nn.Linear(last_input_dim, self.output_dim))
        last_activation = TorchUtils.get_activation_fn(self.output_activation)
        if last_activation is not None:
            model.append(last_activation)
        return nn.Sequential(*model).to(self.device)

    def forward(self, state):
        transformed_state = TorchUtils.transform_input(state)
        return self.model(transformed_state)

    def get_action(self, state):
        ac_logits = self.forward(state)
        ac_probs = F.softmax(ac_logits, dim=0)
        ac_dist = Categorical(ac_probs)
        ac = ac_dist.sample()
        return np.array(ac), ac_probs[ac]

    def get_entropy(self, state):
        ac_logits = self.forward(state)
        ac_probs = F.softmax(ac_logits, dim=0)
        ac_dist = Categorical(ac_probs)
        return ac_dist.entropy()


class ContinuousMLPPolicyNetwork(AbstractPolicyNetwork):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractPolicyNetwork.__init__(input_dim, output_dim, network_setting, device)

    def build_network(self):
        pass

    def get_action(self, state):
        pass

    def get_actionprob(self, state):
        pass

    def get_entropy(self, state):
        pass


class DiscreteCNNPolicyNetwork(AbstractPolicyNetwork):
    pass


class ContinuousCNNPolicyNetwork(AbstractPolicyNetwork):
    pass
