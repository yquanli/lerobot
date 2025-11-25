#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from typing import ClassVar
import math

@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    """
    Configuration for the Piper Leader teleoperator.
    """
    # Additional configuration parameters can be added here as needed
    # 转换常数
    """
    self.robot.GetArmJointCtrl()的单位是1e-3度，JNT_MSGS_TO_RAD将单位从1e-3度转化为1弧度
    self.robot.JointCtrl()的单位是1e-3度，JNT_RAD_TO_MSGS将单位从1弧度转化为1e-3度
    """
    MDEGREE_TO_RAD: ClassVar[float] = (math.pi / 180.0) / 1000.0
    RAD_TO_MDEGREE: ClassVar[float] = (180.0 / math.pi) * 1000.0
    MM_TO_M: ClassVar[float] = 0.001
    M_TO_MM: ClassVar[float] = 1000.0