import glob
import pickle as pkl
import lcm
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from a1_gym_deploy.utils.deployment_runner import DeploymentRunner
from a1_gym_deploy.envs.lcm_agent import LCMAgent
from a1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from a1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def load_and_run_policy(label, experiment_name, probe_policy_label=None, max_vel=1.0, max_yaw_vel=1.0,
                        max_vel_probe=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]
    print(logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
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


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]


def load_policy(logdir):
    body_engine_file_path = logdir + '/engines/body_model.engine'
    adaptation_module_engine_file_path = logdir + '/engines/adaptation_module_model.engine'

    body_engine = load_engine(body_engine_file_path)
    adaptation_module_engine = load_engine(adaptation_module_engine_file_path)

    body_context = body_engine.create_execution_context()
    adaptation_module_context = adaptation_module_engine.create_execution_context()

    body_inputs, body_outputs, body_bindings, body_stream = allocate_buffers(body_engine)
    adaptation_module_inputs, adaptation_module_outputs, adaptation_module_bindings, adaptation_module_stream = allocate_buffers(adaptation_module_engine)

    def policy(obs, info):
        obs_history = obs["obs_history"]  # Assuming obs["obs_history"] is already a numpy array

        # Run adaptation module
        np.copyto(adaptation_module_inputs[0][0], obs_history.ravel())
        latent = do_inference(adaptation_module_context, adaptation_module_bindings, adaptation_module_inputs, adaptation_module_outputs, adaptation_module_stream)[0]

        # Concatenate obs_history and latent
        obs_latent = np.concatenate((obs_history, latent), axis=-1)

        # Run body model
        np.copyto(body_inputs[0][0], obs_latent.ravel())
        action = do_inference(body_context, body_bindings, body_inputs, body_outputs, body_stream)[0]

        info['latent'] = latent  # Use numpy array for latent
        return action  # Return numpy array for action

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretain-a1/train"

    probe_policy_label = None

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, probe_policy_label=probe_policy_label, max_vel=3.0,
                        max_yaw_vel=5.0, max_vel_probe=1.0)

