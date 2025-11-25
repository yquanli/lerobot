"""
比较两个策略配置文件的差异

Usage:
    python myscripts/debug/compare_configs.py \
        --config1 lerobot/smolvla_base \
        --config2 outputs/train/piper_transfer_cube_to_bin/checkpoints/last/pretrained_model
        --config1_name "Base Model" \
        --config2_name "Trained Model" \
        --debug
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import lerobot.policies  # noqa: F401

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.config_validator import compare_policy_configs


def debug_print_config(config, name):
    """打印配置的详细信息用于调试"""
    print(f"\n{'='*80}")
    print(f"🔍 Debug: {name} Configuration Details")
    print(f"{'='*80}")
    
    print(f"\n📊 Input Features:")
    for key, feature in config.input_features.items():
        print(f"  {key}:")
        print(f"    type: {feature.type}")
        print(f"    shape: {feature.shape}")
        if hasattr(feature, 'dtype'):
            print(f"    dtype: {feature.dtype}")
    
    print(f"\n📤 Output Features:")
    for key, feature in config.output_features.items():
        print(f"  {key}:")
        print(f"    type: {feature.type}")
        print(f"    shape: {feature.shape}")
        if hasattr(feature, 'dtype'):
            print(f"    dtype: {feature.dtype}")
    
    print(f"\n⚙️  Other Attributes:")
    important_attrs = [
        'n_obs_steps', 'chunk_size', 'n_action_steps',
        'device', 'use_amp', 'normalization_mapping',
        'max_state_dim', 'max_action_dim',
    ]
    for attr in important_attrs:
        if hasattr(config, attr):
            print(f"  {attr}: {getattr(config, attr)}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two policy configurations")
    parser.add_argument(
        "--config1",
        type=str,
        required=True,
        help="Path to first policy config"
    )
    parser.add_argument(
        "--config2",
        type=str,
        required=True,
        help="Path to second policy config"
    )
    parser.add_argument(
        "--config1_name",
        type=str,
        default="Config 1",
        help="Display name for first config"
    )
    parser.add_argument(
        "--config2_name",
        type=str,
        default="Config 2",
        help="Display name for second config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    try:
        # 加载两个配置
        print(f"\n🔄 Loading configurations...")
        config1 = PreTrainedConfig.from_pretrained(args.config1)
        print(f"✓ Loaded {args.config1_name} from: {args.config1}")
        
        config2 = PreTrainedConfig.from_pretrained(args.config2)
        print(f"✓ Loaded {args.config2_name} from: {args.config2}")
        
        # 如果开启 debug 模式，打印详细信息
        if args.debug:
            debug_print_config(config1, args.config1_name)
            debug_print_config(config2, args.config2_name)
        
        # 比较配置
        differences = compare_policy_configs(
            config1=config1,
            config2=config2,
            config1_name=args.config1_name,
            config2_name=args.config2_name,
            verbose=True,
        )
        
        # 判断是否有差异
        has_differences = any(differences.values())
        
        if has_differences:
            print("\n⚠️  Found differences between configurations")
            print("\n📝 Summary of differences:")
            for category, diffs in differences.items():
                if diffs:
                    print(f"\n  {category}:")
                    for key, diff in diffs.items():
                        print(f"    • {key}: {diff}")
            sys.exit(1)
        else:
            print("\n✅ Configurations are identical")
            sys.exit(0)
        
    except Exception as e:
        logging.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()