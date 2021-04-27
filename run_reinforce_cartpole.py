import gym

from network.policy_network_factory import PolicyNetworkFactory
from common.network_setting import MLPNetworkSetting
from common.torch_utils import TorchUtils
from algorithm.reinforce_algorithm import REINFORCEAlgorithm


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    input_space = env.observation_space
    output_space = env.action_space
    policy_network_factory = PolicyNetworkFactory()
    network_setting = MLPNetworkSetting()
    policy_network = policy_network_factory.get_network(input_space, output_space,
                                                        network_setting, TorchUtils.get_device())
    algo = REINFORCEAlgorithm(env, policy_network, 0.99, 0.001)
    algo.train(1000)

    env.close()
