from network.value_network import (
    DiscreteCNNValueNetwork,
    DiscreteMLPValueNetwork,
    ContinuousCNNValueNetwork,
    ContinuousMLPValueNetwork,
)


class ValueNetworkFactory:
    def get_network(self, input_space, output_space, network_setting, device):
        if len(input_space.shape) == 1:
            if "Discrete" in str(type(output_space)):
                return DiscreteMLPValueNetwork(
                    input_space.shape[0], 1, network_setting, device
                )
            elif "Box" in str(type(output_space)):
                return ContinuousMLPValueNetwork(
                    input_space.shape[0], 1, network_setting, device
                )
        elif len(input_space) == 3:
            if "Discrete" in str(type(output_space)):
                return DiscreteCNNValueNetwork(
                    input_space.shape, 1, network_setting, device
                )
            elif "Box" in str(type(output_space)):
                return ContinuousCNNValueNetwork(
                    input_space.shape, 1, network_setting, device
                )
