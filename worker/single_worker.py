class SingleWorker:
    def __init__(self, env, policy_network, buffer=None, value_network=None):
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network
        self.buffer = buffer

    def sample_trajectory(self, sample_size=-1, rendering=False):
        obs = []
        acs = []
        ac_logprobs = []
        rews = []
        nobs = []
        dones = []
        values = []

        ob = self.env.reset()

        current_t = 0

        while True:
            ac, ac_logprob = self.policy_network.get_action(ob)

            nob, rew, done, _ = self.env.step(ac)
            if rendering:
                self.env.render()

            obs.append(ob)
            acs.append(ac)
            ac_logprobs.append(ac_logprob)
            rews.append(rew)
            nobs.append(nob)
            dones.append(done)
            if self.value_network is not None:
                value = self.value_network(ob)
                values.append(value)
            else:
                values.append(None)

            current_t += 1
            ob = nob

            if sample_size != -1 and current_t >= sample_size:
                break
            if done:
                if sample_size == -1:
                    break
                else:
                    ob = self.env.reset()
        return obs, acs, ac_logprobs, rews, nobs, dones, values
