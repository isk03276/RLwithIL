import torch
from torch.nn import functional as F


class RLLossFunctions:
    @classmethod
    def estimate_pg_loss(cls, ac_logprobs: torch.Tensor, values: torch.Tensor):
        loss = -ac_logprobs * values
        return loss.mean()

    @classmethod
    def estimate_td_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        loss = F.smooth_l1_loss(values, returns)
        return loss.mean()


class ILLossFunction:
    @classmethod
    def estimate_bc_loss(
        cls, acs: torch.Tensor, demo_acs: torch.Tensor
    ) -> torch.Tensor:
        loss = F.MSELoss(acs, demo_acs)
        return loss.mean()
