from multiprocessing import Process

from worker.base_worker import BaseWorker


class MultiWorker(BaseWorker):
    def __init__(
        self, envs, policy_network, num_worker, buffer=None, value_network=None
    ):
        super().__init__(policy_network, buffer, value_network)
        self.envs = envs
        self.num_worker = num_worker

        self.process_list = Process(self.num_worker)
        assert len(self.envs) == self.num_worker

    def sample_trajectory(self, sample_size=-1, rendering=False):
        ob = self.env.reset()

        current_t = 0

        while True:
            ac, ac_logprob = self.policy_network.get_action(ob)

            nob, rew, done, _ = self.env.step(ac)
            if rendering:
                self.env.render()

            self.obs.append(ob)
            self.acs.append(ac)
            self.ac_logprobs.append(ac_logprob)
            self.rews.append(rew)
            self.nobs.append(nob)
            self.dones.append(done)

            if self.value_network is not None:
                value = self.value_network(ob)
                self.values.append(value)
            else:
                self.values.append(None)

            current_t += 1
            ob = nob

            if sample_size != -1 and current_t >= sample_size:
                break
            if done:
                if sample_size == -1:
                    break
                else:
                    ob = self.env.reset()
        return obs, acs, ac_logprobs, rews, nobs, dones, values, rets
