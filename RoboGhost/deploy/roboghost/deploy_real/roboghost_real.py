import sys
sys.path.append('/home/deepcyber-mk/Documents/unitree_rl_gym')
sys.path.append('/home/deepcyber-mk/Documents/unitree_rl_gym/deploy/deploy_real/common')

from typing import Union
import numpy as np
import time
import torch
import onnxruntime
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
import onnxruntime as ort
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, transform_pelvis_to_torso_complete
from common.remote_controller import RemoteController, KeyMap
from config import Config
sys.path.append("/home/roboghost/unitree_rl_gym/")
from roboghost.get_motion_latent import get_motion_latent

joint_seq =['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 
 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 
 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 
 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 
 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 
 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 
 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
joint_xml = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint",  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",  "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint",     "left_shoulder_roll_joint",     "left_shoulder_yaw_joint",
    "left_elbow_joint",     "left_wrist_roll_joint",    "left_wrist_pitch_joint",    "left_wrist_yaw_joint",    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",    "right_shoulder_yaw_joint",    "right_elbow_joint",    "right_wrist_roll_joint",    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"]


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.load_motion_latent = get_motion_latent()
        self.motion_ids = torch.tensor([68], device=self.load_motion_latent._device)
        self.policy =  onnxruntime.InferenceSession(config.policy_path)
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.timestep = 0
        self.history_actor =  np.zeros((4, 90), dtype=np.float32)
        self.action_buffer = np.zeros((self.config.num_actions,), dtype=np.float32)
        self.dof_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                        12, 13, 14, 
                        15, 16, 17, 18, 19, 20, 21, 
                        22, 23, 24, 25, 26, 27, 28]
        self.base_ang_vel_his = np.zeros((15, 3), dtype=np.float32)
        self.joint_pos_his = np.zeros((15, 29), dtype=np.float32)
        self.joint_vel_his = np.zeros((15, 29), dtype=np.float32)
        self.action_his = np.zeros((15, 29), dtype=np.float32)
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.stiffness
        kds = self.config.damping
        # default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        default_pos = self.config.default_angles.copy()
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size): 
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]*5
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i+12]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i+12]*3
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i+12]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            # quat = self.low_state.imu_state.quaternion
            # print("quat",quat)
            time.sleep(self.config.control_dt)

    def init_history_buffers_with_step0(self, history_len=15):
        if isinstance(self.low_state.imu_state.gyroscope, list):
            base_ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        else:

            base_ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        
        if len(base_ang_vel) != 3:
            print(f"Warning: gyroscope data has unexpected length {len(base_ang_vel)}, expected 3")
            if len(base_ang_vel) > 3:
                base_ang_vel = base_ang_vel[:3]
            else:
                base_ang_vel = np.pad(base_ang_vel, (0, 3 - len(base_ang_vel)), 'constant')
        
        for i in range(len(self.dof_idx)):
            self.qj[i] = self.low_state.motor_state[self.dof_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.dof_idx[i]].dq
        
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qpos_urdf = qj_obs
        qj_obs_seq = np.array([qpos_urdf[joint_xml.index(joint)] for joint in joint_seq])
        joint_pos_diff = qj_obs_seq - self.config.default_angles_seq
        
        qvel_urdf = dqj_obs
        dqj_obs_seq = np.array([qvel_urdf[joint_xml.index(joint)] for joint in joint_seq])

        self.base_ang_vel_his = np.tile(base_ang_vel.reshape(1, -1), (history_len, 1))
        self.joint_pos_his = np.tile(joint_pos_diff[None, :23].reshape(1, -1), (history_len, 1))
        self.joint_vel_his = np.tile(dqj_obs_seq[None, :23].reshape(1, -1), (history_len, 1))
        self.action_his = np.zeros((history_len, 23), dtype=np.float32)
        
        
        return {
            'base_ang_vel_his': self.base_ang_vel_his,
            'joint_pos_his': self.joint_pos_his,
            'joint_vel_his': self.joint_vel_his,
            'action_his': self.action_his
        }
    
    def expand_23d_to_29d(self, actions_23d):
        batch_size = actions_23d.shape[0]
        device = actions_23d.device
        
        actions_29d = torch.zeros(batch_size, 29, device=device)
        actions_29d[:, :23] = actions_23d[:, :]
        return actions_29d

    def run(self):
        start = time.time()
        self.counter += 1
        for i in range(len(self.dof_idx)):
            self.qj[i] = self.low_state.motor_state[self.dof_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.dof_idx[i]].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
        if self.config.imu_type == "pelvis":
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_roll = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[1]].q
            waist_pitch= self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[2]].q
            quat_torso = transform_pelvis_to_torso_complete(waist_yaw, waist_roll, waist_pitch, quat)
        
        if self.timestep>=499:
            self.timestep = 0

        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = qj_obs

        motion_latent = self.load_motion_latent.get_motion_latent(self.motion_ids, self.timestep)
        qpos_urdf = qj_obs
        qj_obs_seq =  np.array([qpos_urdf[joint_xml.index(joint)] for joint in joint_seq])

        qvel_urdf = dqj_obs
        dqj_obs_seq =  np.array([qvel_urdf[joint_xml.index(joint)] for joint in joint_seq])

        self.base_ang_vel_his[:-1] = self.base_ang_vel_his[1:]
        self.joint_pos_his[:-1] = self.joint_pos_his[1:]
        self.joint_vel_his[:-1] = self.joint_vel_his[1:]

        self.base_ang_vel_his[-1] = ang_vel
        self.joint_pos_his[-1] = (qj_obs_seq - self.config.default_angles_seq)[:23]
        self.joint_vel_his[-1] = dqj_obs_seq[:23]


        offset = 0
        self.obs[offset:offset + 64] = motion_latent
        offset += 64
        self.obs[offset:offset + 15 * 3] = self.base_ang_vel_his.reshape(-1)
        offset += 15 * 3
        self.obs[offset:offset + 15 * 23] = self.joint_pos_his.reshape(-1)
        offset += 15 * 23
        self.obs[offset:offset + 15 * 23] = self.joint_vel_his.reshape(-1)
        offset += 15 * 23
        self.obs[offset:offset + 15 * 23] = self.action_his.reshape(-1)

        obs_tensor = self.obs.reshape(1, -1).astype(np.float32)
        action = self.policy.run(None, {'input': obs_tensor})[0]
        action = torch.from_numpy(action)  
        action = self.expand_23d_to_29d(action)
        
        action = np.asarray(action).reshape(-1)
        self.action = action.copy()
        self.action_buffer = action.copy()
        self.action_his[:-1] = self.action_his[1:]
        self.action_his[-1] = self.action_buffer[:23]


        target_dof_pos = self.config.default_angles_seq + self.action * self.config.action_scale_seq
        target_dof_pos = target_dof_pos.reshape(-1,)
        target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
        self.timestep += 1
        print("target_dof_pos", target_dof_pos)

        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i+12]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i+12]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i+12]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)
        stop = time.time()
        dt = stop - start
        time.sleep(max(0.02 - dt, 0))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()

    # Load config
    # config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config_path = '/home/roboghost/unitree_rl_gym/roboghost/deploy_real/configs/roboghost.yaml'
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    controller.zero_torque_state()

    controller.move_to_default_pos()

    controller.default_pos_state()
    controller.init_history_buffers_with_step0(history_len=15)
    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")