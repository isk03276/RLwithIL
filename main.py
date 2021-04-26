import gym
from network.policy_network_factory import PolicyNetworkFactory


class Test:
    def env_test(self):
        return gym.make("CartPole-v1")

    def policy_network_test(self):
        env = self.env_test()
        input_space = env.observation_space
        output_space = env.action_space
        policy_network_factory = PolicyNetworkFactory()
        network_setting = (3, 100, )
        policy_network = PolicyNetworkFactory.get_network(input_space, output_space, network_setting)


if __name__ == "__main__":
    test = Test()
    test.policy_network_test()