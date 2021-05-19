import torch.optim as optim

from algorithm.reinforcement_learning.base_rl_algorithm import BaseRLAlgorithm
from algorithm.loss_functions import RLLossFunctions
from worker.multi_worker import MultiWorker
from common.logger import TensorboardLogger
from common.torch_utils import TorchUtils
from common.rl_utils import RLUtils


class A2CAlgorithm(BaseRLAlgorithm):
    def __init__(self, env, policy_network, value_network, gamma,
                 lr, epoch, n_step):
        super().__init__(policy_network, value_network)

        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epoch = epoch
        self.n_step = n_step

        self.policy_network_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)

        self.set_worker(MultiWorker(self.env, self.policy_network, value_network=self.value_network))
        self.logger = TensorboardLogger(str(self), self.env.unwrapped.spec.id)

    def train(self, max_training_step, timesteps_per_learning):
        self.policy_network.train()
        self.value_network.train()


        for training_step in range(max_training_step):
            trajectory = self.worker.sample_trajectory(timesteps_per_learning, False)
            obs, acs, ac_logprobs, rews, nobs, dones, values, next_values = trajectory

            rets = RLUtils.get_nstep_td_return(rews, next_values, dones, self.gamma, self.n_step)

            vf_loss = self.estimate_value_loss(values, rets)
            TorchUtils.update_network(self.value_network_optimizer, vf_loss)

            pg_loss = self.estimate_policy_loss(ac_logprobs, values, rets)
            TorchUtils.update_network(self.policy_network_optimizerloss, pg_loss)

            print(sum(rews))
            self.logger.log("episodic reward", sum(rews), training_step)
            self.logger.log("policy loss", pg_loss, training_step)

    def estimate_policy_loss(self, ac_logprobs, values, rets):
        advantages = RLUtils.get_advantage(values, rets)
        return RLLossFunctions.estimate_pg_loss(ac_logprobs, advantages)

    def estimate_value_loss(self, values, returns):
        RLLossFunctions.estimate_td_loss(values, returns)

    def __str__(self):
        return "A2C"
