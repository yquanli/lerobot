#!/usr/bin/env python

from piper_sdk import C_PiperInterface_V2

_shared_sdk_instance = None

def get_piper_sdk_instance():
    """
    获取或创建 C_PiperInterface_V2 的全局唯一实例。
    这确保了在同一程序中，所有与 Piper 硬件交互的部分
    都通过同一个SDK对象，避免资源冲突。
    """
    global _shared_sdk_instance
    if _shared_sdk_instance is None:
        _shared_sdk_instance = C_PiperInterface_V2("can_0204")
    return _shared_sdk_instance