#!/usr/bin/env python
import logging
from typing import Any

# 导入piper实例管理函数
from lerobot.common.robots.piper_follower.piper_utils import get_piper_sdk_instance

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig
from piper_sdk import C_PiperInterface_V2

logger = logging.getLogger(__name__)

#TODO:精细实现与验证

class PiperLeader(Teleoperator):
    """
    Piper Leader Arm designed by AgileX.
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        # 使用共享实例
        self.robot = get_piper_sdk_instance()
    
    @property
    def action_features(self) -> dict[str, type]:
        return {
        'joint_1': float,
        'joint_2': float,
        'joint_3': float,
        'joint_4': float,
        'joint_5': float,
        'joint_6': float,
        'gripper': float,
    }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        # raise NotImplementedError(f"{self} does not support connect method")
        pass

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self) -> None:
        # raise NotImplementedError(f"{self} does not support calibrate method")
        pass
    
    def configure(self) -> None:
        # raise NotImplementedError(f"{self} does not support configure method")
        pass
    
    def get_action(self) -> dict[str, Any]:
        jnt_action_raw = self.robot.GetArmJointCtrl()
        gripper_action_raw = self.robot.GetArmGripperCtrl()
        action_dict = {
            'joint_1': jnt_action_raw.joint_ctrl.joint_1 * 1e-3,
            'joint_2': jnt_action_raw.joint_ctrl.joint_2 * 1e-3,
            'joint_3': jnt_action_raw.joint_ctrl.joint_3 * 1e-3,
            'joint_4': jnt_action_raw.joint_ctrl.joint_4 * 1e-3,
            'joint_5': jnt_action_raw.joint_ctrl.joint_5 * 1e-3,
            'joint_6': jnt_action_raw.joint_ctrl.joint_6 * 1e-3,
            'gripper': gripper_action_raw.gripper_ctrl.grippers_angle * 1e-3,
        }
        return action_dict
    
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return NotImplementedError
    
    def disconnect(self) -> None:
        # raise NotImplementedError(f"{self} does not support disconnect method")
        pass