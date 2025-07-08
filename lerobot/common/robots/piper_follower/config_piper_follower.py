# lerobot/common/robots/piper_follower/config_piper_follower.py

from dataclasses import dataclass, field


from lerobot.common.cameras import CameraConfig

from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from lerobot.common.robots.config import RobotConfig

from typing import ClassVar
import math

@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):

    #HACK:感觉piper不需要端口号，先这样写着。
    port: str | None = None
    name: str = "piper_follower"


    # 新增 control_mode 字段，默认为 "teleop"
    control_mode: str = "teleop"
    
    #cameras 使用 RealSenseCameraConfig 实例化
    cameras: dict[str, RealSenseCameraConfig] = field(default_factory=lambda: {
        "wrist": RealSenseCameraConfig(
            serial_number_or_name="137322074822",  # 腕部的d435i相机，可以用find_cameras()获取
            fps = 30,
            width = 640,
            height = 480,
            use_depth = False,
        ),
        "front": RealSenseCameraConfig(
            serial_number_or_name="137322078934",  # 正前方的d435i相机，可以用find_cameras()获取
            fps = 30,
            width = 640,
            height = 480,
            use_depth = False,
        )
    })
    
    # 转换常数
    """
    self.robot.GetArmJointCtrl()的单位是1e-3度，JNT_MSGS_TO_RAD将单位从1e-3度转化为1弧度
    self.robot.JointCtrl()的单位是1e-3度，JNT_RAD_TO_MSGS将单位从1弧度转化为1e-3度
    """
    MDEGREE_TO_RAD: ClassVar[float] = (math.pi / 180.0) / 1000.0
    RAD_TO_MDEGREE: ClassVar[float] = (180.0 / math.pi) * 1000.0
    MM_TO_M: ClassVar[float] = 0.001
    M_TO_MM: ClassVar[float] = 1000.0