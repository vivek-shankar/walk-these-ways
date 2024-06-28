import glob
import pickle as pkl
import lcm
import sys
import onnxruntime as ort
import numpy as np

from a1_gym_deploy.utils.deployment_runner import DeploymentRunner
from a1_gym_deploy.envs.lcm_agent import LCMAgent
from a1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from a1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def load_and_run_policy(label, experiment_name, probe_policy_label=None, max_vel=1.0, max_yaw_vel=1.0,
                        max_vel_probe=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/gait-conditioned-agility/pretrain-a1/train/081719.645276")
    logdir = sorted(dirs)[0]
    print(logdir)

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

    print('Config successfully loaded!')

    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6,
                                          yaw_scale=max_yaw_vel, probe_vel_multiplier=(max_vel_probe / max_vel))

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from a1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)
    print('Agent successfully created!')

    policy = load_policy(logdir)
    print('Policy successfully loaded!')
    print(se.get_gravity_vector())

    if probe_policy_label is not None:
        # load agent
        dirs = glob.glob(f"../runs/{probe_policy_label}_*")
        probe_policy_logdir = sorted(dirs)[0]
        with open(probe_policy_logdir + "/parameters.pkl", 'rb') as file:
            probe_cfg = pkl.load(file)
            probe_cfg = probe_cfg["Cfg"]
        probe_policy = load_policy(probe_policy_logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    if probe_policy_label is not None:
        deployment_runner.add_probe_policy(probe_policy, probe_cfg)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)


def load_policy(logdir):
    body_sess = ort.InferenceSession(logdir + '/onnx_models/body_model.onnx')
    adaptation_module_sess = ort.InferenceSession(logdir + '/onnx_models/adaptation_module_model.onnx')


    def policy(obs, info):
        obs_history = obs["obs_history"] #.cpu().numpy()
        print("inputs ", obs_history.shape)
        obs_history = np.array(obs_history, dtype=np.float32)
        priv_obs = obs["privileged_obs"]
        priv_obs = np.array(priv_obs, dtype=np.float32)

        print("privilaged obs ", priv_obs.shape," hist shape ", obs_history.shape)

        # Run adaptation module
        latent = adaptation_module_sess.run(None, {'input': obs_history})[0]

        # Concatenate obs_history and latent
        obs_latent = np.concatenate((obs_history, latent), axis=-1)

        # Run body model
        action = body_sess.run(None, {'input': obs_latent})[0]

        info['latent'] = latent #torch.tensor(latent).to('cpu')
        return action #torch.tensor(action).to('cpu')

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretain-a1/train"

    probe_policy_label = None

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, probe_policy_label=probe_policy_label, max_vel=3.0,
                        max_yaw_vel=5.0, max_vel_probe=1.0)
