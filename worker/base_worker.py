class BaseWorker:
    def __init__(self, policy_network, buffer, value_network):
        self.policy_network = policy_network
        self.value_network = value_network
        self.buffer = buffer
        