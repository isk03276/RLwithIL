import gym
import argparse

from network.policy_network_factory import PolicyNetworkFactory
from common.network_setting import MLPNetworkSetting
from common.torch_utils import TorchUtils
from common.il_utils import ILUtils
from algorithm.reinforcement_learning.reinforce_algorithm import REINFORCEAlgorithm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reinforce algorithm for gym env")
    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--policy-load-path", type=str, default="model/REINFORCECartPole-v0")
    parser.add_argument("--demo-save-path", type=str, default="demodata/")
    parser.add_argument("--num-traj", type=int, default=100)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    input_space = env.observation_space
    output_space = env.action_space
    policy_network_factory = PolicyNetworkFactory()
    network_setting = MLPNetworkSetting()
    policy_network = policy_network_factory.get_network(input_space, output_space,
                                                        network_setting, TorchUtils.get_device())
    policy_network = TorchUtils.load_model(policy_network, args.policy_load_path)
    save_path = args.demo_save_path + str(env)
    ILUtils.collect_demodata_from_model(env, policy_network, args.num_traj, save_path)

    env.close()
