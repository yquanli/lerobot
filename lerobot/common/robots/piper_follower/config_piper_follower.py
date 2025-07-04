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
        "realsense": RealSenseCameraConfig(
            serial_number_or_name="137322074822",  # 腕部的d435i相机，可以用find_cameras()获取
            fps = 30,
            width = 640,
            height = 480,
            use_depth = False,
        )
    })
    
    # # cameras
    # cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # 转换常数
    RAD_TO_SDK_UNITS: ClassVar[float] = (180.0 / math.pi) * 1000.0
    SDK_UNITS_TO_RAD: ClassVar[float] = (math.pi / 180.0) / 1000.0
    MM_TO_SDK_UNITS: ClassVar[float] = 1000.0
    SDK_UNITS_TO_MM: ClassVar[float] = 0.001