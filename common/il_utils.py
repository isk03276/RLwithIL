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
            trajectory = worker.sample_trajectory(-1, rendering=True)
            obs, acs, ac_logprobs, rews, nobs, dones, values, next_values = trajectory
            demo_obs.append(np.array(obs))
            demo_acs.append(np.array(acs))

        np.savez(save_path, obs=np.array(demo_obs), acs=np.array(demo_acs))
        


            
