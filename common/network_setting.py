class MLPNetworkSetting:
    def __init__(self, hidden_units=(100, 100), hidden_activations=("relu", "relu"), output_activation="linear"):
        assert len(hidden_units) == len(hidden_activations)

        self.hidden_units = hidden_units
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation


class CNNPolicyNetworkSetting:
    pass
