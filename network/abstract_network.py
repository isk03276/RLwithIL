import abc
import torch.nn as nn

from common.torch_utils import TorchUtils


class AbstractNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, network_setting, device):
        super(AbstractNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = network_setting.hidden_units
        self.hidden_activations = network_setting.hidden_activations
        self.output_activation = network_setting.output_activation
        self.num_layers = len(self.hidden_units)
        self.device = device

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