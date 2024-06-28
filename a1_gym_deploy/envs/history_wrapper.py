import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


class HistoryWrapper:
    def __init__(self, env):
        self.env = env

        if isinstance(self.env.cfg, dict):
            self.obs_history_length = self.env.cfg["env"]["num_observation_history"]
        else:
            self.obs_history_length = self.env.cfg.env.num_observation_history
        self.num_obs_history = self.obs_history_length * self.env.num_obs
        self.num_privileged_obs = self.env.num_privileged_obs

        # Allocate memory on GPU for obs_history
        self.obs_history = np.zeros((self.env.num_envs, self.num_obs_history), dtype=np.float32)
        self.obs_history_gpu = cuda.mem_alloc(self.obs_history.nbytes)
        cuda.memcpy_htod(self.obs_history_gpu, self.obs_history)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        # Update obs_history on GPU
        self._update_obs_history_gpu(obs)

        return {'obs': obs, 'privileged_obs': privileged_obs,
                'obs_history': self._get_obs_history_from_gpu()}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()

        # Update obs_history on GPU
        self._update_obs_history_gpu(obs)

        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self._get_obs_history_from_gpu()}

    def get_obs(self):
        obs = self.env.get_obs()
        privileged_obs = self.env.get_privileged_observations()

        # Update obs_history on GPU
        self._update_obs_history_gpu(obs)

        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self._get_obs_history_from_gpu()}

    def reset_idx(self, env_ids):
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        cuda.memcpy_htod(self.obs_history_gpu, self.obs_history)
        return ret

    def reset(self):
        ret = self.env.reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        cuda.memcpy_htod(self.obs_history_gpu, self.obs_history)
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

    def _update_obs_history_gpu(self, obs):
        obs_host = self.obs_history[:, self.env.num_obs:]
        obs_host = np.concatenate((obs_host, obs), axis=-1)
        cuda.memcpy_htod(self.obs_history_gpu, obs_host)

    def _get_obs_history_from_gpu(self):
        obs_history_host = np.empty_like(self.obs_history)
        cuda.memcpy_dtoh(obs_history_host, self.obs_history_gpu)
        return obs_history_host

    def __getattr__(self, name):
        return getattr(self.env, name)
