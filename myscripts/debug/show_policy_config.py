"""
以直观易读的方式显示策略配置的所有细节

Usage:
    # 查看预训练模型配置
    python myscripts/debug/show_policy_config.py --policy_path lerobot/smolvla_base
    
    # 查看训练后的模型配置
    python myscripts/debug/show_policy_config.py \
        --policy_path outputs/train/piper/checkpoints/last/pretrained_model
    
    # 保存到文件
    python myscripts/debug/show_policy_config.py \
        --policy_path lerobot/smolvla_base \
        --output config_details.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import lerobot.policies  # noqa: F401

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType


def format_value(value: Any, indent: int = 0) -> str:
    """格式化值以便打印"""
    prefix = "  " * indent
    
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        elif len(value) <= 5 and all(isinstance(x, (int, float)) for x in value):
            return str(list(value))
        else:
            items = "\n".join(f"{prefix}  - {item}" for item in value)
            return f"\n{items}"
    elif isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        items = "\n".join(f"{prefix}  {k}: {format_value(v, indent+1)}" for k, v in value.items())
        return f"\n{items}"
    elif value is None:
        return "None"
    elif isinstance(value, FeatureType):
        return value.name
    else:
        return str(value)


def print_section(title: str, width: int = 80):
    """打印节标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str, width: int = 80):
    """打印子节标题"""
    print(f"\n{title}")
    print("-" * width)


def show_policy_config(policy_path: str, output_file: str = None):
    """显示策略配置的所有细节"""
    
    # 重定向输出（如果指定了输出文件）
    original_stdout = sys.stdout
    if output_file:
        sys.stdout = open(output_file, 'w', encoding='utf-8')
    
    try:
        # 加载配置
        print(f"Loading policy configuration from: {policy_path}")
        config = PreTrainedConfig.from_pretrained(policy_path)
        
        # ========================================
        # 1. 基本信息
        # ========================================
        print_section("📋 BASIC INFORMATION")
        print(f"  Policy Type:          {config.type}")
        print(f"  Device:               {config.device}")
        print(f"  Mixed Precision:      {config.use_amp}")
        
        if hasattr(config, 'pretrained_path'):
            print(f"  Pretrained Path:      {config.pretrained_path}")
        
        # ========================================
        # 2. 输入特征（Observations）
        # ========================================
        print_section("📊 INPUT FEATURES (Observations)")
        
        if hasattr(config, 'input_features') and config.input_features:
            for key, feature in config.input_features.items():
                print_subsection(f"  {key}")
                print(f"    Type:               {feature.type.name if isinstance(feature.type, FeatureType) else feature.type}")
                print(f"    Shape:              {feature.shape}")
                
                if hasattr(feature, 'dtype'):
                    print(f"    Data Type:          {feature.dtype}")
                
                if hasattr(feature, 'normalization'):
                    print(f"    Normalization:      {feature.normalization}")
        else:
            print("  (No input features defined)")
        
        # ========================================
        # 3. 输出特征（Actions）
        # ========================================
        print_section("📤 OUTPUT FEATURES (Actions)")
        
        if hasattr(config, 'output_features') and config.output_features:
            for key, feature in config.output_features.items():
                print_subsection(f"  {key}")
                print(f"    Type:               {feature.type.name if isinstance(feature.type, FeatureType) else feature.type}")
                print(f"    Shape:              {feature.shape}")
                
                if hasattr(feature, 'dtype'):
                    print(f"    Data Type:          {feature.dtype}")
                
                if hasattr(feature, 'normalization'):
                    print(f"    Normalization:      {feature.normalization}")
        else:
            print("  (No output features defined)")
        
        # ========================================
        # 4. 归一化配置
        # ========================================
        print_section("🔧 NORMALIZATION CONFIGURATION")
        
        if hasattr(config, 'normalization_mapping'):
            print("  Normalization Mapping:")
            for feat_type, norm_type in config.normalization_mapping.items():
                feat_name = feat_type.name if isinstance(feat_type, FeatureType) else str(feat_type)
                print(f"    {feat_name:20s} → {norm_type}")
        
        if hasattr(config, 'normalization_statistics'):
            print("\n  Normalization Statistics:")
            if config.normalization_statistics:
                for key, stats in config.normalization_statistics.items():
                    print(f"    {key}:")
                    if isinstance(stats, dict):
                        for stat_name, stat_value in stats.items():
                            if isinstance(stat_value, (list, tuple)) and len(stat_value) <= 10:
                                print(f"      {stat_name:10s}: {stat_value}")
                            else:
                                print(f"      {stat_name:10s}: {type(stat_value).__name__} (shape: {getattr(stat_value, 'shape', 'N/A')})")
            else:
                print("    (No statistics available)")
        
        # ========================================
        # 5. 时间配置
        # ========================================
        print_section("⏱️  TEMPORAL CONFIGURATION")
        
        temporal_attrs = {
            'n_obs_steps': 'Observation Steps',
            'chunk_size': 'Prediction Chunk Size',
            'n_action_steps': 'Action Execution Steps',
        }
        
        for attr, label in temporal_attrs.items():
            if hasattr(config, attr):
                print(f"  {label:30s}: {getattr(config, attr)}")
        
        # ========================================
        # 6. 图像处理配置
        # ========================================
        print_section("🖼️  IMAGE PROCESSING")
        
        image_attrs = {
            'resize_imgs_with_padding': 'Resize with Padding',
            'add_empty_images': 'Add Empty Images',
            'add_image_special_tokens': 'Add Image Special Tokens',
        }
        
        for attr, label in image_attrs.items():
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"  {label:30s}: {value}")
        
        # ========================================
        # 7. 模型架构配置
        # ========================================
        print_section("🏗️  MODEL ARCHITECTURE")
        
        if hasattr(config, 'vlm_model_name'):
            print(f"  VLM Model Name:       {config.vlm_model_name}")
        
        architecture_attrs = {
            'max_state_dim': 'Max State Dimension',
            'max_action_dim': 'Max Action Dimension',
            'action_head_hidden_dim': 'Action Head Hidden Dim',
            'action_head_hidden_layers': 'Action Head Hidden Layers',
            'action_head_dropout': 'Action Head Dropout',
        }
        
        for attr, label in architecture_attrs.items():
            if hasattr(config, attr):
                print(f"  {label:30s}: {getattr(config, attr)}")
        
        # ========================================
        # 8. 训练配置
        # ========================================
        print_section("🎯 TRAINING CONFIGURATION")
        
        training_attrs = {
            'load_vlm_weights': 'Load VLM Weights',
            'freeze_vision_encoder': 'Freeze Vision Encoder',
            'train_expert_only': 'Train Expert Only',
            'train_state_proj': 'Train State Projection',
        }
        
        for attr, label in training_attrs.items():
            if hasattr(config, attr):
                value = getattr(config, attr)
                emoji = "✅" if value else "❌"
                print(f"  {label:30s}: {emoji} {value}")
        
        # ========================================
        # 9. 优化器配置
        # ========================================
        print_section("📈 OPTIMIZER CONFIGURATION")
        
        optimizer_attrs = {
            'optimizer_lr': 'Learning Rate',
            'optimizer_betas': 'Optimizer Betas',
            'optimizer_eps': 'Optimizer Epsilon',
            'optimizer_weight_decay': 'Weight Decay',
            'grad_clip_norm': 'Gradient Clip Norm',
        }
        
        for attr, label in optimizer_attrs.items():
            if hasattr(config, attr):
                print(f"  {label:30s}: {getattr(config, attr)}")
        
        # ========================================
        # 10. 其他配置
        # ========================================
        print_section("⚙️  OTHER CONFIGURATION")
        
        # 收集所有未显示的属性
        displayed_attrs = set()
        for section in [temporal_attrs, image_attrs, architecture_attrs, training_attrs, optimizer_attrs]:
            displayed_attrs.update(section.keys())
        
        displayed_attrs.update([
            'type', 'device', 'use_amp', 'pretrained_path',
            'input_features', 'output_features',
            'normalization_mapping', 'normalization_statistics',
            'vlm_model_name',
        ])
        
        other_attrs = {}
        for attr in dir(config):
            if not attr.startswith('_') and attr not in displayed_attrs:
                if not callable(getattr(config, attr)):
                    other_attrs[attr] = getattr(config, attr)
        
        if other_attrs:
            for attr, value in sorted(other_attrs.items()):
                # 跳过一些内部属性
                if attr in ['from_pretrained', 'push_to_hub', 'save_pretrained', 'to_dict', 'to_json_string']:
                    continue
                
                formatted_value = format_value(value)
                if '\n' in formatted_value:
                    print(f"\n  {attr}:{formatted_value}")
                else:
                    print(f"  {attr:30s}: {formatted_value}")
        
        # ========================================
        # 11. 配置摘要
        # ========================================
        print_section("📝 CONFIGURATION SUMMARY")
        
        # 统计信息
        num_input_features = len(config.input_features) if hasattr(config, 'input_features') else 0
        num_output_features = len(config.output_features) if hasattr(config, 'output_features') else 0
        
        print(f"  Total Input Features:   {num_input_features}")
        print(f"  Total Output Features:  {num_output_features}")
        
        # 计算输入维度
        total_state_dim = 0
        total_images = 0
        
        if hasattr(config, 'input_features'):
            for key, feature in config.input_features.items():
                if feature.type == FeatureType.STATE:
                    total_state_dim += feature.shape[0] if feature.shape else 0
                elif feature.type == FeatureType.VISUAL:
                    total_images += 1
        
        print(f"  Total State Dimension:  {total_state_dim}")
        print(f"  Total Image Inputs:     {total_images}")
        
        # 计算输出维度
        total_action_dim = 0
        if hasattr(config, 'output_features'):
            for key, feature in config.output_features.items():
                if feature.type == FeatureType.ACTION:
                    total_action_dim += feature.shape[0] if feature.shape else 0
        
        print(f"  Total Action Dimension: {total_action_dim}")
        
        # 训练策略摘要
        print("\n  Training Strategy:")
        if hasattr(config, 'load_vlm_weights') and config.load_vlm_weights:
            print("    ✅ Using pretrained VLM weights")
            if hasattr(config, 'freeze_vision_encoder') and config.freeze_vision_encoder:
                print("    ❄️  Vision encoder frozen")
            else:
                print("    🔥 Vision encoder trainable")
            
            if hasattr(config, 'train_expert_only') and config.train_expert_only:
                print("    🎯 Training Action Expert only")
            else:
                print("    🎯 Training entire model")
        else:
            print("    ⚠️  Training from scratch (no pretrained weights)")
        
        print("\n" + "=" * 80)
        print("✅ Configuration display completed")
        print("=" * 80 + "\n")
        
    finally:
        # 恢复标准输出
        if output_file:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"✅ Configuration saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Display policy configuration in a readable format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View pretrained model config
  python myscripts/debug/show_policy_config.py --policy_path lerobot/smolvla_base
  
  # View trained model config
  python myscripts/debug/show_policy_config.py \\
      --policy_path outputs/train/piper/checkpoints/last/pretrained_model
  
  # Save to file
  python myscripts/debug/show_policy_config.py \\
      --policy_path lerobot/smolvla_base \\
      --output smolvla_config.txt
        """
    )
    
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to policy configuration (HuggingFace repo or local path)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, prints to stdout if not specified)"
    )
    
    args = parser.parse_args()
    
    try:
        show_policy_config(args.policy_path, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()