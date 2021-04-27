import numpy as np
import torch

from common.torch_utils import TorchUtils


class RLUtils:
    @classmethod
    def get_return(cls, rews, gamma):
        returns = [rews[-1]]
        for i in range(1, len(rews)):
            returns.append(rews[len(rews)-1-i] + gamma*returns[i-1])
        returns.reverse()
        return torch.Tensor(returns)
