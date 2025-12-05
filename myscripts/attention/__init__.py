"""
SmolVLA Attention Visualization Module

用于可视化和分析 SmolVLA 模型中的注意力模式。

目录结构:
    myscripts/attention/
    ├── inputs/         # 放置输入图片
    ├── outputs/        # 输出可视化结果
    │   └── ep000_frame0050/
    │       ├── vlm_attention/       # VLM 层注意力
    │       ├── denoising_attention/ # Denoising 交叉注意力
    │       ├── cross_camera/        # 跨相机注意力矩阵
    │       ├── attention_statistics.json
    │       └── input_info.txt
    ├── __init__.py
    ├── visualize_attention.py
    └── run_visualization.py

使用方法:
    # 使用数据集（推荐）
    python run_visualization.py --repo-id Sprinng/piper_transfer_cube_to_bin --episode-index 0 --frame-index 50
    
    # 使用本地图片
    python run_visualization.py --image cube.png --instruction "Pick up the cube"

SmolVLA 架构说明:
    - VLM 层 (0-15): 视觉语言理解，自注意力
    - Denoising 层: Action Expert 交叉注意力，查询 VLM 的 KV cache
"""

from .visualize_attention import (
    SmolVLAAttentionVisualizer,
    AttentionConfig,
    SequenceLayout,
)

__all__ = [
    "SmolVLAAttentionVisualizer",
    "AttentionConfig",
    "SequenceLayout",
]