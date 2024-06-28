import time
import lcm
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from a1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    def __init__(self, cfg, se, command_profile):
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        self.command_profile = command_profile

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.timestep = 0

        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_envs = 1
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_commands = self.cfg["commands"]["num_commands"]
        self.device = 'cuda:0'

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.commands_scale = np.array(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"], self.obs_scales["body_height_cmd"], 1, 1, 1, 1, 1,
             self.obs_scales["footswing_height_cmd"], self.obs_scales["body_pitch_cmd"],
             # 0, self.obs_scales["body_pitch_cmd"],
             self.obs_scales["body_roll_cmd"], self.obs_scales["stance_width_cmd"],
             self.obs_scales["stance_length_cmd"], self.obs_scales["aux_reward_cmd"], 1, 1, 1, 1, 1, 1
             ])[:self.num_commands]


        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]
        self.default_dof_pos = np.array([self.cfg["init_state"]["default_joint_angles"][name] for name in joint_names])
        try:
            self.default_dof_pos_scale = np.array([self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"]])
        except KeyError:
            self.default_dof_pos_scale = np.ones(12)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros((1, self.num_commands), dtype=np.float32)
        self.actions = np.zeros(12, dtype=np.float32)
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.gravity_vector = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(12, dtype=np.float32)
        self.dof_vel = np.zeros(12, dtype=np.float32)
        self.body_linear_vel = np.zeros(3, dtype=np.float32)
        self.body_angular_vel = np.zeros(3, dtype=np.float32)
        self.joint_pos_target = np.zeros(12, dtype=np.float32)
        self.joint_vel_target = np.zeros(12, dtype=np.float32)
        self.torques = np.zeros(12, dtype=np.float32)
        self.contact_state = np.ones(4, dtype=np.float32)

        self.joint_idxs = self.se.joint_idxs

        self.gait_indices = np.zeros(self.num_envs, dtype=np.float32)
        self.clock_inputs = np.zeros((self.num_envs, 4), dtype=np.float32)

        # Allocate GPU memory for numpy arrays
        self.commands_gpu = cuda.mem_alloc(self.commands.nbytes)
        self.actions_gpu = cuda.mem_alloc(self.actions.nbytes)
        self.last_actions_gpu = cuda.mem_alloc(self.last_actions.nbytes)
        self.gravity_vector_gpu = cuda.mem_alloc(self.gravity_vector.nbytes)
        self.dof_pos_gpu = cuda.mem_alloc(self.dof_pos.nbytes)
        self.dof_vel_gpu = cuda.mem_alloc(self.dof_vel.nbytes)
        self.body_linear_vel_gpu = cuda.mem_alloc(self.body_linear_vel.nbytes)
        self.body_angular_vel_gpu = cuda.mem_alloc(self.body_angular_vel.nbytes)
        self.joint_pos_target_gpu = cuda.mem_alloc(self.joint_pos_target.nbytes)
        self.joint_vel_target_gpu = cuda.mem_alloc(self.joint_vel_target.nbytes)
        self.torques_gpu = cuda.mem_alloc(self.torques.nbytes)
        self.contact_state_gpu = cuda.mem_alloc(self.contact_state.nbytes)
        self.gait_indices_gpu = cuda.mem_alloc(self.gait_indices.nbytes)
        self.clock_inputs_gpu = cuda.mem_alloc(self.clock_inputs.nbytes)

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.is_currently_probing = False

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_obs(self):

        self.gravity_vector = self.se.get_gravity_vector()
        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)
        self.commands[:, :] = cmds[:self.num_commands]
        if reset_timer:
            self.reset_gait_indices()
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        self.body_linear_vel = self.se.get_body_linear_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()

        # Move numpy data to GPU
        cuda.memcpy_htod(self.commands_gpu, self.commands)
        cuda.memcpy_htod(self.dof_pos_gpu, self.dof_pos)
        cuda.memcpy_htod(self.dof_vel_gpu, self.dof_vel)
        cuda.memcpy_htod(self.body_linear_vel_gpu, self.body_linear_vel)
        cuda.memcpy_htod(self.body_angular_vel_gpu, self.body_angular_vel)
        cuda.memcpy_htod(self.gravity_vector_gpu, self.gravity_vector)
        cuda.memcpy_htod(self.actions_gpu, self.actions)
        cuda.memcpy_htod(self.last_actions_gpu, self.last_actions)

        ob = np.concatenate((self.gravity_vector.reshape(1, -1),
                             self.commands * self.commands_scale,
                             (self.dof_pos - self.default_dof_pos).reshape(1, -1) * self.obs_scales["dof_pos"],
                             self.dof_vel.reshape(1, -1) * self.obs_scales["dof_vel"],
                             np.clip(self.actions, -self.cfg["normalization"]["clip_actions"],
                                        self.cfg["normalization"]["clip_actions"]).reshape(1, -1)
                             ), axis=1)

        if self.cfg["env"]["observe_two_prev_actions"]:
            ob = np.concatenate((ob,
                            self.last_actions.reshape(1, -1)), axis=1)

        if self.cfg["env"]["observe_clock_inputs"]:
            ob = np.concatenate((ob, self.clock_inputs.reshape(1, -1)), axis=1)

        if self.cfg["env"]["observe_vel"]:
            ob = np.concatenate(
                (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
                 self.body_angular_vel.reshape(1, -1) * self.obs_scales["ang_vel"],
                 ob), axis=1)

        if self.cfg["env"]["observe_only_lin_vel"]:
            ob = np.concatenate(
                (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
                 ob), axis=1)

        if self.cfg["env"]["observe_yaw"]:
            heading = self.se.get_yaw()
            ob = np.concatenate((ob, heading.reshape(1, -1)), axis=-1)

        self.contact_state = self.se.get_contact_state()
        if "observe_contact_states" in self.cfg["env"].keys() and self.cfg["env"]["observe_contact_states"]:
            ob = np.concatenate((ob, self.contact_state.reshape(1, -1)), axis=-1)

        if "terrain" in self.cfg.keys() and self.cfg["terrain"]["measure_heights"]:
            robot_height = 0.25
            self.measured_heights = np.zeros(
                (len(self.cfg["terrain"]["measured_points_x"]), len(self.cfg["terrain"]["measured_points_y"]))).reshape(
                1, -1)
            heights = np.clip(robot_height - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales["height_measurements"]
            ob = np.concatenate((ob, heights.reshape(1, -1)), axis=1)

        return np.array(ob)

    def get_privileged_observations(self):
        return None

    def publish_action(self, action, hard_reset=False):

        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12] * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains

        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = np.zeros(12, dtype=np.float32)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_gait_indices(self):
        self.gait_indices = np.zeros(self.num_envs, dtype=np.float32)

    def step(self, actions, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = np.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz')
        self.time = time.time()
        obs = self.get_obs()

        # clock accounting
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        if self.num_commands == 8:
            bounds = 0
            durations = self.commands[:, 7]
        else:
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
        self.gait_indices = np.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + phases]
        else:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + bounds,
                                 self.gait_indices + phases]
        self.clock_inputs[:, 0] = np.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[:, 1] = np.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[:, 2] = np.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[:, 3] = np.sin(2 * np.pi * self.foot_indices[3])

        # Move clock inputs to GPU
        cuda.memcpy_htod(self.clock_inputs_gpu, self.clock_inputs)

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                 }

        self.timestep += 1
        return obs, None, None, infos
