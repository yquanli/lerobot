"""
训练前的完整验证检查

Usage:
    python scripts/debug/validate_training_setup.py \
        --policy_path lerobot/smolvla_base \
        --dataset_repo_id Sprinng/piper_transfer_cube_to_bin \
        --output_dir outputs/train/piper_transfer_cube_to_bin \
        --rename_map '{"observation.images.top_rgb":"observation.images.camera1", "observation.images.wrist_rgb":"observation.images.camera2", "observation.images.side_rgb":"observation.images.camera3"}'
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import lerobot.policies  # noqa: F401

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.config_validator import (
    validate_policy_dataset_compatibility,
    print_dataset_statistics,
)


def parse_rename_map(rename_map_str: str) -> dict:
    """解析命令行传入的 rename_map JSON 字符串"""
    if not rename_map_str:
        return None
    
    rename_map_str = rename_map_str.replace("'", '"')
    return json.loads(rename_map_str)


def check_output_dir(output_dir: Path) -> dict:
    """检查输出目录状态"""
    info = {
        'exists': output_dir.exists(),
        'has_checkpoints': False,
        'latest_checkpoint': None,
    }
    
    if info['exists']:
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.exists():
            info['has_checkpoints'] = True
            last_checkpoint = checkpoints_dir / "last" / "pretrained_model"
            if last_checkpoint.exists():
                info['latest_checkpoint'] = str(last_checkpoint)
    
    return info


def main():
    parser = argparse.ArgumentParser(description="Validate entire training setup before starting")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rename_map", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    all_passed = True
    
    print("\n" + "=" * 80)
    print("🔍 Training Setup Validation")
    print("=" * 80)
    
    try:
        # 1. 检查输出目录
        print("\n📁 Checking output directory...")
        output_dir = Path(args.output_dir)
        dir_info = check_output_dir(output_dir)
        
        if dir_info['exists']:
            print(f"  ⚠️  Output directory already exists: {output_dir}")
            if dir_info['has_checkpoints']:
                print(f"  ⚠️  Found existing checkpoints")
                if dir_info['latest_checkpoint']:
                    print(f"  📦 Latest checkpoint: {dir_info['latest_checkpoint']}")
                print("  ⚠️  Training will resume from existing checkpoint or overwrite")
        else:
            print(f"  ✓ Output directory will be created: {output_dir}")
        
        # 2. 加载策略配置
        print(f"\n🤖 Loading policy configuration...")
        rename_map = parse_rename_map(args.rename_map) if args.rename_map else None
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        print(f"  ✓ Policy type: {policy_cfg.type}")
        print(f"  ✓ Device: {policy_cfg.device}")
        print(f"  ✓ Mixed precision: {policy_cfg.use_amp}")
        
        # 3. 加载数据集
        print(f"\n📊 Loading dataset...")
        dataset = LeRobotDataset(args.dataset_repo_id)
        print(f"  ✓ Total frames: {len(dataset)}")
        print(f"  ✓ Total episodes: {dataset.num_episodes}")
        print(f"  ✓ FPS: {dataset.meta.fps}")
        
        # 4. 估算训练资源
        print(f"\n⚡ Estimating training resources...")
        total_frames = len(dataset)
        batch_size = args.batch_size
        steps_per_epoch = total_frames // batch_size
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Num workers: {args.num_workers}")
        
        # 5. 打印数据集统计
        print_dataset_statistics(dataset.meta, verbose=True)
        
        # 6. 执行兼容性检查
        result = validate_policy_dataset_compatibility(
            policy_cfg=policy_cfg,
            dataset_meta=dataset.meta,
            rename_map=rename_map,
            raise_on_error=False,
            verbose=True,
        )
        
        if not result['passed']:
            all_passed = False
        
        # 7. 汇总结果
        print("\n" + "=" * 80)
        print("📋 Validation Summary")
        print("=" * 80)
        
        if all_passed and result['passed']:
            print("✅ All checks passed! Ready to start training.")
            print(f"\nSuggested command:")
            print(f"lerobot-train \\")
            print(f"  --dataset.repo_id={args.dataset_repo_id} \\")
            print(f"  --policy.path={args.policy_path} \\")
            print(f"  --output_dir={args.output_dir} \\")
            if rename_map:
                print(f"  --rename_map='{args.rename_map}' \\")
            print(f"  --batch_size={args.batch_size}")
        else:
            print("❌ Validation FAILED! Please fix the issues above before training.")
        
        print("=" * 80 + "\n")
        
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        logging.error(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()