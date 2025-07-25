#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any
import numpy as np
import torch
import threading
# 导入 LeRobot 基类和相机、机器人设备工具
from ..robot import Robot
from lerobot.common.cameras.utils import make_cameras_from_configs
#from lerobot.common.cameras.realsense import RealSenseCamera
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# 导入piper实例管理函数
from .piper_utils import get_piper_sdk_instance

# 导入我们之前定义好的配置类
from lerobot.common.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.common.robots.piper_follower.xense_data  import Xense
logger = logging.getLogger(__name__)

class PiperFollower(Robot):
    """
    Piper Follower Arm designed by AgileX.
    """

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        # 1. 初始化机械臂 SDK
        self.config = config
        self.robot = get_piper_sdk_instance()  # 获取全局唯一的 Piper SDK 实例
        
        # 2. 初始化相机
        # 根据配置实例化相机对象
        self.cameras = make_cameras_from_configs(config.cameras)

        # 3. 初始化触觉传感器
        self.xense_0 = Xense(device_id="0G000205")  # 初始化 Xense 传感器
        self.xense_1 = Xense(device_id="0G000206")  # 初始化第二个 Xense 传感器
        


        
        # 机械臂物理参数
        self.num_joints = 6
        self.gripper_range_mm = [0.0, 70.0]  # 物理开合范围
        
        # 根据控制模式决定是否使能
        if self.config.control_mode == "policy":
            logger.info("Control mode is 'policy', enabling Piper follower.")
            # 使能piper follower
            self.robot.EnablePiper()
        else:
            logger.info("Control mode is 'teleop', no extra enabling needed for Piper follower.")

    
    # @property
    # def _cameras_ft(self) -> dict[str, tuple]:
    #     return {
    #         cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
    #     }

    @property
    def observation_features(self) -> dict:
        """
        定义观测数据的结构和类型。
        """
        # 定义电机位置和末端执行器姿态的特征
        motor_and_pose_features = {
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "joint_6": float,
            "gripper": float,
            "end_pose.x": float,
            "end_pose.y": float,
            "end_pose.z": float,
            "end_pose.roll": float,
            "end_pose.pitch": float,
            "end_pose.yaw": float,
        }

        # 定义相机的特征
        camera_features = {}
        for cam_name, cam_config in self.config.cameras.items():
            # RGB 图像特征
            camera_features[f"{cam_name}_rgb"] = (cam_config.height, cam_config.width, 3)
            # 深度图像特征
            if cam_config.use_depth:
                camera_features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width, 1)
                
        # 定义 Xense 传感器的特征
        xense_features = {
            "xense_0_rectify": (700, 400, 3),
            "xense_0_diff": (700, 400, 3),
            "xense_0_depth": (700, 400, 1),
            "xense_0_force": (35, 20, 3),
            "xense_0_force_norm": (35, 20, 3),
            "xense_0_force_resultant": (6,),
            "xense_0_mesh_init": (35, 20, 3),
            "xense_0_mesh_now": (35, 20, 3),
            "xense_0_mesh_flow": (35, 20, 3),
            "xense_1_rectify": (700, 400, 3),
            "xense_1_diff": (700, 400, 3),
            "xense_1_depth": (700, 400, 1),
            "xense_1_force": (35, 20, 3),
            "xense_1_force_norm": (35, 20, 3),
            "xense_1_force_resultant": (6,),
            "xense_1_mesh_init": (35, 20, 3),
            "xense_1_mesh_now": (35, 20, 3),
            "xense_1_mesh_flow": (35, 20, 3),
        }

        # 合并所有特征到一个字典中
        return {**motor_and_pose_features, **camera_features,**xense_features}

    @property
    def action_features(self) -> dict:
        """
        定义动作指令的结构和类型。
        """
        # 动作指令是每个关节和夹爪的目标位置
        return {
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "joint_6": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        """
        检查机械臂和相机是否已连接。
        """
        return True and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:

        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.robot.ConnectPort()
        
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True  # piper已使用上位机校准，此处不需要再校准了
    
    def calibrate(self) -> None:
        """
        由于 Piper 已经在上位机上校准，此处不需要实际操作。
        """
        pass
    
    def configure(self) -> None:
        """
        将任何一次性或运行时配置应用于 robot。
        这可能包括设置电机参数、控制模式或初始状态。
        """
        # 目前看来没什么要执行的
        raise NotImplementedError("Piper Follower does not require configuration in code.")


             
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # 读取piper的关节状态、夹爪状态和endpose
        start = time.perf_counter()
        jnt_state_raw = self.robot.GetArmJointMsgs().joint_state 
        gripper_state_raw = self.robot.GetArmGripperMsgs().gripper_state
        end_pose_raw = self.robot.GetArmEndPoseMsgs().end_pose
        obs_dict = {
            "joint_1": jnt_state_raw.joint_1,
            "joint_2": jnt_state_raw.joint_2,
            "joint_3": jnt_state_raw.joint_3,   
            "joint_4": jnt_state_raw.joint_4,
            "joint_5": jnt_state_raw.joint_5,
            "joint_6": jnt_state_raw.joint_6,
            "gripper": gripper_state_raw.grippers_angle,
            "end_pose.x": end_pose_raw.X_axis,
            "end_pose.y": end_pose_raw.Y_axis,
            "end_pose.z": end_pose_raw.Z_axis, #0.001mm
            "end_pose.roll": end_pose_raw.RX_axis,
            "end_pose.pitch": end_pose_raw.RY_axis,
            "end_pose.yaw": end_pose_raw.RZ_axis #0.001degrees 
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joint_state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            #读取rgb图像
            obs_dict[f"{cam_key}_rgb"] = cam.async_read()
            if cam.use_depth:
                depth_image = cam.read_depth()  # 同步读取深度图像
                depth_image = depth_image[:,:,np.newaxis]  # 添加通道维度
                obs_dict[f"{cam_key}_depth"] = depth_image #TODO：跟着gemini改成异步深度读取
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        
        self.xense_0.run()
        self.xense_1.run()
        (
        xense_0_diff,
        xense_0_rectify,
        xense_0_depth,
        xense_0_force,
        xense_0_force_norm,
        xense_0_force_resultant,
        xense_0_mesh_init,
        xense_0_mesh_now,
        xense_0_mesh_flow
        ) = self.xense_0.read_data()

        (
        xense_1_diff,
        xense_1_rectify,
        xense_1_depth,
        xense_1_force,
        xense_1_force_norm,
        xense_1_force_resultant,
        xense_1_mesh_init,
        xense_1_mesh_now,
        xense_1_mesh_flow
        ) = self.xense_1.read_data()

        obs_dict["xense_0_rectify"] = xense_0_rectify  # (700, 400, 3)
        obs_dict["xense_0_diff"] = xense_0_diff  # (700, 400, 3)
        obs_dict["xense_0_depth"] = xense_0_depth  # (700, 400, 1)
        obs_dict["xense_0_force"] = xense_0_force  # (35, 20, 3)
        obs_dict["xense_0_force_norm"] = xense_0_force_norm  # (35, 20, 3)
        obs_dict["xense_0_force_resultant"] = xense_0_force_resultant  # (6,)
        obs_dict["xense_0_mesh_init"] = xense_0_mesh_init  # (35, 20, 3)
        obs_dict["xense_0_mesh_now"] = xense_0_mesh_now    # (35, 20, 3)
        obs_dict["xense_0_mesh_flow"] = xense_0_mesh_flow  # (35, 20, 3)
        
        obs_dict["xense_1_rectify"] = xense_1_rectify  # (700, 400, 3)
        obs_dict["xense_1_diff"] = xense_1_diff  # (700, 400, 3)
        obs_dict["xense_1_depth"] = xense_1_depth  # (700, 400, 1)
        obs_dict["xense_1_force"] = xense_1_force  # (35, 20, 3)
        obs_dict["xense_1_force_norm"] = xense_1_force_norm  # (35, 20, 3)
        obs_dict["xense_1_force_resultant"] = xense_1_force_resultant  # (6,)
        obs_dict["xense_1_mesh_init"] = xense_1_mesh_init  # (35, 20, 3)
        obs_dict["xense_1_mesh_now"] = xense_1_mesh_now    # (35, 20, 3)
        obs_dict["xense_1_mesh_flow"] = xense_1_mesh_flow  # (35, 20, 3)

        with open("observation.txt", "w") as f:
            for key, value in obs_dict.items():
                 f.write(f"{key}: {np.array(value).tolist()}\n")
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Ensure the action dictionary contains the required keys
        required_keys = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
        for key in required_keys:
            if key not in action:
                raise ValueError(f"Missing required key '{key}' in action dictionary.")
        
        def get_float_value(value) -> float:
            if isinstance(value,torch.Tensor):
                return value.item()
            return value

        # Convert joint values to SDK units
        jnt_1 = round(get_float_value(action['joint_1']) * PiperFollowerConfig.RAD_TO_MDEGREE)
        jnt_2 = round(get_float_value(action['joint_2']) * PiperFollowerConfig.RAD_TO_MDEGREE)
        jnt_3 = round(get_float_value(action['joint_3']) * PiperFollowerConfig.RAD_TO_MDEGREE)
        jnt_4 = round(get_float_value(action['joint_4']) * PiperFollowerConfig.RAD_TO_MDEGREE)
        jnt_5 = round(get_float_value(action['joint_5']) * PiperFollowerConfig.RAD_TO_MDEGREE)
        jnt_6 = round(get_float_value(action['joint_6']) * PiperFollowerConfig.RAD_TO_MDEGREE)
        
        # Convert gripper value to SDK units
        gripper_angle = round(get_float_value(action['gripper']) * PiperFollowerConfig.M_TO_MM)
        
        # Send the action to the robot
        self.robot.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.robot.JointCtrl(jnt_1, jnt_2, jnt_3, jnt_4, jnt_5, jnt_6)
        self.robot.GripperCtrl(abs(gripper_angle), 1000, 0x01, 0)
        
        logger.info(f"{self} sent action: {action}")
        return action #TODO：安全范围界定

    def back_to_zero(self):
        # Back to the home position
        self.robot.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.robot.JointCtrl(0, 0, 0, 0, 0, 0)
        self.robot.GripperCtrl(0, 1000, 0x01, 0)
        print("Piper is back to zero.")
        return None

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # 机械臂回到零位
        self.back_to_zero()

        # self.robot.DisablePiper()
        self.robot.DisconnectPort()
        for cam in self.cameras.values():
            cam.disconnect()


        logger.info(f"{self} disconnected.")