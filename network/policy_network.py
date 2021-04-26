from network.abstract_policy_network import AbstractPolicyNetwork


class DiscreteMLPPolicyNetwork(AbstractPolicyNetwork):
    def __init__(self, input_dim, output_dim, network_setting):
        AbstractPolicyNetwork.__init__(input_dim, output_dim, network_setting)

    def build_network(self):
        pass

    def get_action(self, state):
        pass

    def get_actionprob(self, state):
        pass

    def get_entropy(self, state):
        pass


class ContinuousMLPPolicyNetwork(AbstractPolicyNetwork):
    def __init__(self, input_dim, output_dim, network_setting):
        AbstractPolicyNetwork.__init__(input_dim, output_dim, network_setting)

    def build_network(self):
        pass

    def get_action(self, state):
        pass

    def get_actionprob(self, state):
        pass

    def get_entropy(self, state):
        pass


class DiscreteCNNPolicyNetwork(AbstractPolicyNetwork):
    pass

class ContinuousCNNPolicyNetwork(AbstractPolicyNetwork):
    pass