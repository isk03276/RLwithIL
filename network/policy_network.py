from network.abstract_policy_network import AbstractPolicyNetwork
from common.torch_utils import TorchUtils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


class DiscreteMLPPolicyNetwork(AbstractPolicyNetwork):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractPolicyNetwork.__init__(self, input_dim, output_dim, network_setting, device)
        self.model = self._build_network()

    def forward(self, state):
        transformed_state = TorchUtils.transform_input(state)
        return self.model(transformed_state)

    def get_action(self, state):
        ac_logits = self.forward(state)
        ac_probs = F.softmax(ac_logits, dim=0)
        ac_dist = Categorical(ac_probs)
        ac = ac_dist.sample()
        return ac.item(), ac_dist.log_prob(ac)

    def get_entropy(self, state):
        ac_logits = self.forward(state)
        ac_probs = F.softmax(ac_logits, dim=0)
        ac_dist = Categorical(ac_probs)
        return ac_dist.entropy()


class ContinuousMLPPolicyNetwork(AbstractPolicyNetwork):
    def __init__(self, input_dim, output_dim, network_setting, device):
        AbstractPolicyNetwork.__init__(self, input_dim, output_dim, network_setting, device)
        self.model = self._build_network()

        self.log_std = np.ones(self.output_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.Tensor(self.log_std))

    def forward(self, state):
        transformed_state = TorchUtils.transform_input(state)
        return self.model(transformed_state)

    def get_action(self, state):
        mu = self.forward(state)
        std = torch.exp(self.log_std)

        ac_dist = Normal(mu, std)
        ac = ac_dist.sample()
        ac_logprob = ac_dist.log_prob(ac).sum()
        return ac, ac_logprob

    def get_entropy(self, state):
        mu = self.forward(state)
        std = torch.exp(self.log_std)

        ac_dist = Normal(mu, std)
        return ac_dist.entropy()


class DiscreteCNNPolicyNetwork(AbstractPolicyNetwork):
    pass


class ContinuousCNNPolicyNetwork(AbstractPolicyNetwork):
    pass
