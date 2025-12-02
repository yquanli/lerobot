"""
SmolVLA Attention Visualization Module

This module provides tools for visualizing and analyzing attention patterns
in the SmolVLA vision-language-action model.

Directory Structure:
    myscripts/attention/
    ├── input/          # Place input images here
    ├── outputs/        # Output visualizations (organized by image name)
    │   ├── cube/       # Outputs for cube.png
    │   ├── scene1/     # Outputs for scene1.jpg
    │   └── ...
    ├── __init__.py
    ├── visualize_attention.py
    ├── analyze_smolvla_attention.py
    └── run_visualization.py

Usage:
    # Process all images in input folder
    python run_visualization.py --all
    
    # Process specific image
    python run_visualization.py --image cube.png
    
    # Detailed analysis
    python run_visualization.py --detailed --image cube.png
"""

from .visualize_attention import (
    SmolVLAAttentionVisualizer,
    AttentionConfig,
)

from .analyze_smolvla_attention import (
    SmolVLADetailedAnalyzer,
    DetailedAttentionConfig,
)

__all__ = [
    "SmolVLAAttentionVisualizer",
    "AttentionConfig", 
    "SmolVLADetailedAnalyzer",
    "DetailedAttentionConfig",
]