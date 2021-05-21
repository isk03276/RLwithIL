import torch.optim as optim
import torch

from algorithm.reinforcement_learning.base_rl_algorithm import BaseRLAlgorithm
from algorithm.loss_functions import RLLossFunctions
from worker.single_worker import SingleWorker
from common.logger import TensorboardLogger
from common.rl_utils import RLUtils
from common.torch_utils import TorchUtils
from buffer.on_policy_buffer import OnPolicyBuffer


class REINFORCEAlgorithm(BaseRLAlgorithm):
    def __init__(self, env, policy_network, gamma, lr, epoch=1):
        super().__init__(env, policy_network, None, lr)

        self.gamma = gamma
        self.epoch = epoch

        self.set_policy_network(policy_network)
        self.set_policy_network_optimizer(optim.Adam(self.policy_network.parameters(), lr=self.lr))
        self.set_worker(SingleWorker(self.env, self.policy_network, OnPolicyBuffer()))

        self.logger = TensorboardLogger(str(self), self.env.spec.id)

    def train(self, max_training_step):
        for training_step in range(max_training_step):
            trajectory = self.worker.sample_trajectory(-1, False)
            obs, acs, ac_logprobs, rews, nobs, dones, values, next_values = trajectory
            loss = self.estimate_policy_loss(ac_logprobs, rews)
            TorchUtils.update_network(self.policy_network_optimizer, loss)

            print(sum(rews), loss)
            self.logger.log("episodic reward", sum(rews), training_step)
            self.logger.log("policy loss", loss, training_step)


    def estimate_policy_loss(self, ac_logprobs, rews):
        returns = RLUtils.get_mc_return(rews, self.gamma)
        return RLLossFunctions.estimate_pg_loss(ac_logprobs, returns)

    def __str__(self):
        return "REINFORCE"

    def estimate_value_loss(self, *args):
        pass

