from typing import List

import numpy as np
from numpy.core.shape_base import _accumulate
import torch

from common.torch_utils import TorchUtils


class RLUtils:
    @classmethod
    def get_mc_return(cls, rews: List[float], gamma: float) -> torch.Tensor:
        """
        Get monte-carlo return.
        R = r(s,a) + gamma*R(s',a')
        """
        returns = [rews[-1]]
        for i in range(1, len(rews)):
            returns.append(rews[len(rews) - 1 - i] + gamma * returns[i - 1])
        returns.reverse()
        return torch.Tensor(returns)

    @classmethod
    def divice_list(cls, dones: List[bool], n_step: int) -> List[bool]:
        """
        Get indexes of divided array for nstep learning
        """

        done_idx_list = list(np.where(np.array(dones) == True)[0])
        result = [0]

        for i in range(len(dones)):
            if i in done_idx_list:
                result.append(i)
            elif i % n_step == n_step - 1:
                result.append(i)
        return result

    @classmethod
    def get_nstep_td_return(
        cls,
        rews: List[float],
        next_values: List[torch.Tensor],
        dones: List[bool],
        gamma: list,
        n_step: int = 1,
    ) -> torch.Tensor:
        """
        Get n-step TD return.
        R = 0 if done(s,a) else v(s')
        R = r(s,a) + gamma*R
        """
        returns = np.zeros_like(rews, dtype=np.float)

        returns[-1] = next_values[-1]
        divided_indexes = cls.divice_list(dones, n_step)
        for i in range(len(rews) - 1, 0, -1):
            if i in divided_indexes:
                returns[i] = rews[i] if dones[i] else next_values[i]
            returns[i - 1] = rews[i - 1] + gamma * returns[i]
        return torch.from_numpy(returns).float()

    @classmethod
    def get_advantage(cls, values: np.ndarray, rets: np.ndarray) -> np.ndarray:
        return values - rets
