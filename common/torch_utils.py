import torch
import torch.nn as nn
import numpy as np


class TorchUtils:
    @classmethod
    def get_activation_fn(cls, activation_fn):
        if activation_fn == "tanh":
            return nn.Tanh
        elif activation_fn == "relu":
            return nn.ReLU
        elif activation_fn == "leakyrelu":
            return nn.LeakyReLU
        elif activation_fn == "linear":
            return None
        else:
            raise NameError

    @classmethod
    def get_device(cls, name="cpu"):
        if name == "cpu":
            return torch.device(name)
        elif name == "auto":
            return torch.device("cuda")

    @classmethod
    def transform_input(cls, input):
        if not isinstance(input, torch.Tensor):
            input = torch.from_numpy(np.array(input)).float()
        return input
