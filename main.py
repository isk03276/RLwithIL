import gym
from network.policy_network_factory import PolicyNetworkFactory
from common.network_setting import MLPNetworkSetting
from common.torch_utils import TorchUtils
from worker.single_worker import SingleWorker


class Test:
    def env_test(self):
        return gym.make("CartPole-v1")

    def policy_network_test(self):
        env = self.env_test()
        input_space = env.observation_space
        output_space = env.action_space
        policy_network_factory = PolicyNetworkFactory()
        network_setting = MLPNetworkSetting()
        policy_network = policy_network_factory.get_network(input_space, output_space,
                                                            network_setting, TorchUtils.get_device())
        state = env.reset()
        print(policy_network.get_action(state))

    def worker_test(self):
        env = self.env_test()
        input_space = env.observation_space
        output_space = env.action_space
        policy_network_factory = PolicyNetworkFactory()
        network_setting = MLPNetworkSetting()
        policy_network = policy_network_factory.get_network(input_space, output_space,
                                                            network_setting, TorchUtils.get_device())
        worker = SingleWorker(env, policy_network)
        print(worker.sample_trajectory(10))

if __name__ == "__main__":
    test = Test()
    test.policy_network_test()
