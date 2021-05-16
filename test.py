import gym
from network.policy_network_factory import PolicyNetworkFactory
from network.value_network_factory import ValueNetworkFactory
from common.network_setting import MLPNetworkSetting
from common.torch_utils import TorchUtils
from worker.single_worker import SingleWorker
from test.test_rl_utils import *

import torch


class Test:
    def env_test(self):
        return gym.make("MountainCarContinuous-v0")

    def policy_network_test(self):
        env = self.env_test()
        input_space = env.observation_space
        output_space = env.action_space
        policy_network_factory = PolicyNetworkFactory()
        network_setting = MLPNetworkSetting()
        policy_network = policy_network_factory.get_network(input_space, output_space,
                                                            network_setting, TorchUtils.get_device())
        state = env.reset()
        ac = policy_network.get_action(state)
        next_state, _, _, _ = env.step(ac)

        input = torch.stack(state, next_state)
        print(policy_network(input))

    def value_network_test(self):
        env = self.env_test()
        input_space = env.observation_space
        output_space = env.action_space
        value_network_factory = ValueNetworkFactory()
        network_setting = MLPNetworkSetting(output_activation="linear")
        value_network = value_network_factory.get_network(input_space, output_space,
                                                            network_setting, TorchUtils.get_device())
        state = env.reset()
        value = value_network.get_value(state)
        print(value)

    def worker_test(self):
        env = self.env_test()
        input_space = env.observation_space
        output_space = env.action_space
        policy_network_factory = PolicyNetworkFactory()
        network_setting = MLPNetworkSetting()
        policy_network = policy_network_factory.get_network(input_space, output_space,
                                                            network_setting, TorchUtils.get_device())
        worker = SingleWorker(env, policy_network)
        print(worker.sample_trajectory(10, False))

    def check_test(self):
        check_td_nstep_return()



if __name__ == "__main__":
    test = Test()
    test.check_test()
