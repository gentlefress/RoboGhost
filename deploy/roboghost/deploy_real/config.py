
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]

            self.stiffness = np.array(config["stiffness"], dtype=np.float32)
            self.damping = np.array(config["damping"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.default_angles_seq = np.array(config["default_angles_seq"], dtype=np.float32)
            self.action_scale_seq = np.array(config["action_scale_seq"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
