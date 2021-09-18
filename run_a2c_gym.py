import gym
import argparse

from network.policy_network_factory import PolicyNetworkFactory
from network.value_network_factory import ValueNetworkFactory
from common.network_setting import MLPNetworkSetting
from common.torch_utils import TorchUtils
from algorithm.reinforcement_learning.a2c_algorithm import A2CAlgorithm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reinforce algorithm for gym env")
    parser.add_argument(
        "--env-name", type=str, default="CartPole-v0"
    )  # MountainCarContinuous-v0
    parser.add_argument(
        "--lr", type=float, help="learning rate of policy network", default=0.001
    )
    parser.add_argument("--gamma", type=float, help="discounted rate", default=0.99)
    parser.add_argument(
        "--timesteps-per-learning", type=float, help="discounted rate", default=128
    )
    parser.add_argument(
        "--max-training-step",
        type=int,
        help="max episode size for learning",
        default=10000,
    )
    parser.add_argument(
        "--epoch", type=int, help="epoch for learning policy network", default=3
    )
    parser.add_argument("--n-step", type=int, help="n-step for td learning", default=4)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    input_space = env.observation_space
    output_space = env.action_space
    network_setting = MLPNetworkSetting()

    policy_network_factory = PolicyNetworkFactory()
    policy_network = policy_network_factory.get_network(
        input_space, output_space, network_setting, TorchUtils.get_device()
    )
    value_network_factory = ValueNetworkFactory()
    value_network = value_network_factory.get_network(
        input_space, output_space, network_setting, TorchUtils.get_device()
    )
    algo = A2CAlgorithm(
        env, policy_network, value_network, args.gamma, args.lr, args.epoch, args.n_step
    )

    algo.train(args.max_training_step, args.timesteps_per_learning)

    # TorchUtils.save_model(policy_network, str(algo), env.unwrapped.spec.id)

    env.close()
