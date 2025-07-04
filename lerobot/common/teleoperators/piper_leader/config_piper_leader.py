#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    """
    Configuration for the Piper Leader teleoperator.
    """
    # Additional configuration parameters can be added here as needed