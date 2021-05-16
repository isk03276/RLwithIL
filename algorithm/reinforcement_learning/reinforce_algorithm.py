from algorithm.reinforcement_learning.abstract_rl_algorithm import AbstractRLAlgorithm
from common.rl_utils import RLUtils
from worker.single_worker import SingleWorker
from common.logger import TensorboardLogger

import torch.optim as optim
import torch


class REINFORCEAlgorithm(AbstractRLAlgorithm):
    def __init__(self, env, policy_network, gamma, lr, epoch):
        AbstractRLAlgorithm.__init__(self, policy_network, None)

        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epoch = epoch

        self.set_policy_network(policy_network)
        self.set_policy_network_optimizer(optim.Adam(self.policy_network.parameters(), lr=self.lr))
        self.worker = SingleWorker(self.env, self.policy_network)

        self.logger = TensorboardLogger(str(self), self.env.spec.id)

    def train(self, max_training_step):
        for training_step in range(max_training_step):
            trajectory = self.worker.sample_trajectory(-1, True)
            obs, acs, ac_logprobs, rews, nobs, dones, values = trajectory

            loss = self.estimate_policy_loss(obs, acs, ac_logprobs, rews, nobs)
            self.optimize_policy_network(loss)

            print(sum(rews), loss)
            self.logger.log("episodic reward", sum(rews), training_step)
            self.logger.log("policy loss", loss, training_step)

    def estimate_policy_loss(self, obs, acs, ac_logprobs, rews, nobs):
        returns = RLUtils.get_return(rews, self.gamma)
        loss = -torch.stack(ac_logprobs) * returns
        return loss.mean()

    def __str__(self):
        return "REINFORCE"

    def estimate_value_loss(self, *args):
        pass

