from worker.base_worker import BaseWorker


class SingleWorker(BaseWorker):
    def __init__(self, env, policy_network, buffer, value_network=None):
        super().__init__(policy_network, buffer, value_network)
        self.env = env

    def sample_trajectory(self, sample_size=-1, rendering=False):
        ob = self.env.reset()

        current_t = 0

        while True:
            ac, ac_logprob = self.policy_network.get_action(ob)

            nob, rew, done, _ = self.env.step(ac)
            if rendering:
                self.env.render()

            if self.value_network is not None:
                value = self.value_network(ob)
                next_value = self.value_network(nob)
            else:
                value = None
                next_value = None

            self.buffer.add(ob, ac, ac_logprob, rew, nob, done, value, next_value)
            current_t += 1
            ob = nob

            if sample_size != -1 and current_t >= sample_size:
                break
            if done:
                if sample_size == -1:
                    break
                else:
                    ob = self.env.reset()
        return self.buffer.sample(sample_size)
