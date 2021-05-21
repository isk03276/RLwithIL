import gym
import argparse

from network.policy_network_factory import PolicyNetworkFactory
from common.network_setting import MLPNetworkSetting
from common.torch_utils import TorchUtils
from algorithm.reinforcement_learning.reinforce_algorithm import REINFORCEAlgorithm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reinforce algorithm for gym env")
    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--lr", type=float, help="learning rate of policy network", default=0.001)
    parser.add_argument("--gamma", type=float, help="discounted rate", default=0.99)
    parser.add_argument("--max-training-step", type=int, help="max episode size for learning", default=1000)
    parser.add_argument("--epoch", type=int, help="epoch for learning policy network", default=1)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    input_space = env.observation_space
    output_space = env.action_space
    policy_network_factory = PolicyNetworkFactory()
    network_setting = MLPNetworkSetting()
    policy_network = policy_network_factory.get_network(input_space, output_space,
                                                        network_setting, TorchUtils.get_device())
    algo = REINFORCEAlgorithm(env, policy_network, args.gamma, args.lr, args.epoch)
    algo.train(args.max_training_step)

    TorchUtils.save_model(policy_network, str(algo), env.unwrapped.spec.id)

    env.close()
