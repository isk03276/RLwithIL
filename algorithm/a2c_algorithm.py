from algorithm.reinforce_algorithm import AbstractRLAlgorithm
from worker.multi_worker import MultiWorker
from common.logger import TensorboardLogger

import torch.optim as optim


class A2CAlgorithm(AbstractRLAlgorithm):
    def __init__(self, env, policy_network, value_network, gamma,
                 lr, epoch):
        super().__init__(policy_network, value_network)

        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epoch = epoch

        self.policy_network_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)

        self.set_worker(MultiWorker(self.env, self.policy_network, value_network=self.value_network))
        self.logger = TensorboardLogger(str(self), self.env.unwrapped.spec.id)

    def train(self, max_training_step, timesteps_per_learning):
        self.policy_network.train()
        self.value_network.train()


        for training_step in range(max_training_step):
            trajectory = self.worker.sample_trajectory(timesteps_per_learning, False)
            obs, acs, ac_logprobs, rews, nobs, dones, values, rets = trajectory

            loss = self.estimate_policy_loss(obs, acs, ac_logprobs, rews, nobs)
            self.optimize_policy_network(loss)

            print(sum(rews), loss)
            self.logger.log("episodic reward", sum(rews), training_step)
            self.logger.log("policy loss", loss, training_step)

    def estimate_policy_loss(self, *args):
        pass

    def estimate_value_loss(self, *args):
        pass

    def __str__(self):
        return "A2C"

