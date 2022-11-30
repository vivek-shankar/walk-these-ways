from a1_utilities.a1_sensor_histories import NormObsWithImg
from a1_utilities.a1_sensor_process import observation_to_joint_position, observation_to_torque, prepare_obs
#from a1_utilities.logger import StateLogger
#from a1_utilities.a1_sensor_histories import VisualHistory
import numpy as np
import torch
import datetime
from PIL import Image as im

class PolicyWrapper():
  def __init__(
    self,
    policy,
    obs_normalizer_mean, obs_normalizer_var,
    get_image_interval,
    save_dir_name, 
    sliding_frames=True, no_tensor=False, 
    default_joint_angle=None,
    action_range=None,
    use_visual=True,
    clip_motor=False,
    clip_motor_value=0.5,
    use_foot_contact=False,
    save_log=False
  ):
    self.pf = policy
    self.no_tensor = no_tensor

    self.get_image_interval = get_image_interval

    self.use_visual = use_visual
    
    if default_joint_angle == None:
      default_joint_angle = [0.0, 0.9,-1.8]
    self.default_joint_angle = np.array(default_joint_angle * 4)

    self.current_joint_angle = default_joint_angle
    self.clip_motor = clip_motor
    self.clip_motor_value = clip_motor_value
    if action_range == None:
      action_range = [0.05, 0.5, 0.5]
    self.action_range = np.array(action_range * 4)

    self.action_lb = self.default_joint_angle - self.action_range
    self.action_ub = self.default_joint_angle + self.action_range

    self.normalizer = NormObsWithImg(46, obs_normalizer_mean, obs_normalizer_var, use_visual=self.use_visual)

    self.count = 0

    
  def process_obs(self, observation, last_action, depth_frame=None, depth_scale=None):
    state = prepare_obs(observation, last_action)
    img_obs = None
    if self.use_visual:
      new_image = torch.from_numpy(-depth_frame * depth_scale).to("cuda:0")
      new_image = torch.nn.functional.interpolate(new_image.view(1,1,240,424), size=(64, 64)).view(64, 64)
      new_image = torch.nan_to_num(new_image, neginf=0)
      new_image = torch.clamp(new_image, min=-8)
      # new_image = 1 + (new_image / torch.min(new_image + 1e-4))
      
      # print(depth_frame)
      cam_img = new_image.cpu().detach().numpy()
      fname = './image/'+str(self.count)+'.png'
      image = im.fromarray(
              cam_img.astype(np.uint8), mode="L"
          )
      image.save(fname)
      self.count += 1
      img_obs = torch.reshape(new_image, (-1,))
    if self.use_visual:
      obs = self.normalizer.observation(state, img_obs)
    else:
      obs = state.view(1,46)
    

    return obs

  def process_act(self, action):
    if self.vis_only:
      return action
    else:
      diagonal_action_normalized = action
      right_act_normalized, left_act_normalized = np.split(diagonal_action_normalized, 2)
      action_normalized = np.concatenate(
          [right_act_normalized, left_act_normalized, left_act_normalized, right_act_normalized]
      )

      action_ub = self.action_ub
      action_lb = self.action_lb
      action = 0.5 * (np.tanh(action_normalized) + 1) * (action_ub - action_lb) + action_lb
      if self.clip_motor:
        action = np.clip(
          action,
          self.current_joint_angle - self.clip_motor_value,
          self.current_joint_angle + self.clip_motor_value
        )
    return action


  def get_action(self, observation, last_action, depth_frame=None, depth_scale=None):
    '''
    This function process raw observation, fed normalized observation into
    the network, de-normalize and output the action.
    '''
    default_dof_pos = torch.Tensor((0.0, 0.9,-1.8, 0.0, 0.9,-1.8, 0.0, 0.9,-1.8, 0.0, 0.9,-1.8)).to('cuda:0')
    ob_t = self.process_obs(observation, last_action, depth_frame, depth_scale).view(1, 46)
    #action = self.pf.eval_act(ob_t)
    action = self.pf(ob_t)
    # action = self.process_act(action)
    # if self.save_log:
    #   self.policy_action_saver.record(action)
    action = action[0][0]
    action = torch.clip(action-default_dof_pos, -1, 1)+default_dof_pos
    return action

  # def write(self):
  #   if self.save_log:
  #     self.ob_tensor_saver.write()
  #     self.policy_action_saver.write()
