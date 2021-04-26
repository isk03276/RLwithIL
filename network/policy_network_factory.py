from network.policy_network import *


class PolicyNetworkFactory:
    def get_network(self, input_space, output_space, network_setting, device):
        if len(input_space.shape) == 1:
            if "Discrete" in str(type(output_space)):
                return DiscreteMLPPolicyNetwork(input_space.shape[0], output_space.n, network_setting, device)
            elif "Box" in str(type(output_space)):
                return ContinuousMLPPolicyNetwork(input_space.shape[0], output_space.shape[0], network_setting, device)
        elif len(input_space) == 3:
            if "Discrete" in str(type(output_space)):
                return DiscreteCNNPolicyNetwork(input_space.shape[0], output_space.shape[0], network_setting, device)
            elif "Box" in str(type(output_space)):
                return ContinuousCNNPolicyNetwork(input_space.shape[0], output_space.n, network_setting, device)


