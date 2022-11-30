import threading
import time
import numpy as np
import torch

from robot_interface import RobotInterface
from a1_utilities.a1_sensor_process import observation_to_joint_position, observation_to_joint_velocity

class RobotController(object):
  def __init__(
    self,
    control_freq=40,
    default_action=None,
    use_high_level_command=False,
    save_log=False,
    log_interval=10,
    state_save_path=None,
  ):
    if default_action is None:
      # no output command for robot
      default_action = np.zeros(60)
      default_high_action = np.zeros(3)

    self.control_freq = control_freq
    self.control_interval = 1 / self.control_freq
    self.action = default_action
    self.observation = None              # observation buffer
    self.control_thread_enabled = False  # when False, thread_main ends
    self.control_thread = None           # control thread
    self.high_action = default_high_action

    self.use_high_level_command = use_high_level_command

    # robot interface
    if self.use_high_level_command:
      self.robot_interface = RobotInterface(0)
    else:
      self.robot_interface = RobotInterface()

    self.save_log = save_log
    self.log_interval = log_interval
    self.state_save_path = state_save_path
    self.observation = self.robot_interface.receive_observation()

  def start_thread(self):
    self.control_thread_enabled = True
    self.control_thread = threading.Thread(target=self.control_function)
    self.control_thread.start()
    print("Control thread started")

  def stop_thread(self):
    self.control_thread_enabled = False
    self.control_thread.join()
    print("Control thread stopped")

  def control_function(self):
    control_step = 0
    while self.control_thread_enabled:
      start_time = time.time()
      # single transcation
      if self.use_high_level_command:
        self.robot_interface.send_high_command(self.high_action)
      else:
        self.robot_interface.send_command(self.action)

      self.observation = self.robot_interface.receive_observation()

      end_time = time.time()
      time.sleep(
          max(0, self.control_interval - (end_time - start_time))
      )
      control_step += 1

  def get_observation(self):
    return self.observation

  def set_action(self, action):
    self.action = action.cpu().detach().numpy()

  def set_high_action(self, high_action):
    self.high_action = high_action

  def set_target_dof_pos(self, target_dof_pos):
    try:
      self.target_dof_pos = target_dof_pos.cpu().detach().numpy()
    except:
      self.target_dof_pos = target_dof_pos
