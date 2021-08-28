import random

import numpy as np

from worker.single_worker import SingleWorker
from buffer.on_policy_buffer import OnPolicyBuffer


class ILUtils:
    @classmethod
    def collect_demodata_from_model(cls, env, network, num_traj, save_path):
        buffer = OnPolicyBuffer()
        worker = SingleWorker(env, network, buffer)

        demo_obs = []
        demo_acs = []

        for ep in range(num_traj):
            trajectory = worker.sample_trajectory(-1, rendering=False)
            obs, acs, ac_logprobs, rews, nobs, dones, values, next_values = trajectory
            demo_obs.append(np.array(obs))
            demo_acs.append(np.array(acs))

        np.savez(save_path, obs=np.array(demo_obs), acs=np.array(demo_acs))


class DemodataManager:
    def __init__(self, demodata_path, mode="random"):
        self.load_demodata(demodata_path)
        self.mode = mode

        if self.mode == "full sequence":
            pass
        elif self.mode == "partial sequence":
            self._flatten_demodata()
        elif self.mode == "random":
            self._shuffle_demodata()
        else:
            raise NotImplementedError

        self.idx = 0
        self.demodata_size = len(self.obs)

    def load_demodata(self, demodata_path):
        demodata = np.load(demodata_path, allow_pickle=True)
        self.obs = demodata["obs"]
        self.acs = demodata["acs"]

    def _flatten_demodata(self):
        self.obs = self._flatten_array(self.obs)
        self.acs = self._flatten_array(self.acs)

    def _flatten_array(self, array: np.ndarray):
        if len(array.shape) == 3:
            return np.vstack(array)
        elif len(array.shape) == 2:
            return np.hstack(array)

    def _shuffle_demodata(self):
        self._flatten_demodata()
        random_idx = list(range(len(self.obs)))
        random.shuffle(random_idx)
        self.obs = self.obs[random_idx]
        self.acs = self.acs[random_idx]

    def sample(self, batch_size, mode="random"):
        assert batch_size < self.demodata_size

        if batch_size == -1:
            return self.obs, self.acs

        if self.idx + batch_size > self.demodata_size:
            new_idx = batch_size - self.demodata_size + self.idx
            _obs = np.concatenate((self.obs[self.idx :], self.obs[:new_idx]), axis=0)
            _acs = np.concatenate((self.acs[self.idx :], self.acs[:new_idx]), axis=0)
        else:
            new_idx = self.idx + batch_size
            _obs = self.obs[self.idx : self.idx + batch_size]
            _acs = self.acs[self.idx : self.idx + batch_size]

        self.idx = new_idx
        return _obs, _acs
