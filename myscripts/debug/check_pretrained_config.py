"""
检查单个预训练模型配置与单个数据集的兼容性
✅ 特征维度匹配检查
✅ 特征类型匹配检查
✅ 归一化配置检查

Usage:
    python scripts/debug/check_pretrained_config.py \
        --policy_path lerobot/smolvla_base \
        --dataset_repo_id Sprinng/piper_transfer_cube_to_bin \
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

# ⭐ 导入 policies 包以注册所有策略配置
# 这会触发 lerobot/policies/__init__.py 中的导入，从而注册所有策略子类
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
    
    # 将单引号替换为双引号以符合 JSON 标准
    rename_map_str = rename_map_str.replace("'", '"')
    return json.loads(rename_map_str)


def main():
    parser = argparse.ArgumentParser(description="Check pretrained policy config compatibility with dataset")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to pretrained policy (e.g., lerobot/smolvla_base)"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., Sprinng/piper_transfer_cube_to_bin)"
    )
    parser.add_argument(
        "--rename_map",
        type=str,
        default=None,
        help='Feature rename mapping as JSON string (e.g., \'{"old_name":"new_name"}\')'
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show dataset statistics"
    )
    parser.add_argument(
        "--raise_on_error",
        action="store_true",
        help="Raise exception if validation fails"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    try:
        # 解析 rename_map
        rename_map = parse_rename_map(args.rename_map) if args.rename_map else None
        
        # 加载策略配置
        print(f"\n🔄 Loading policy configuration from: {args.policy_path}")
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        print(f"✓ Policy type: {policy_cfg.type}")
        
        # 加载数据集
        print(f"\n🔄 Loading dataset: {args.dataset_repo_id}")
        dataset = LeRobotDataset(args.dataset_repo_id)
        print(f"✓ Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")
        
        # 打印数据集统计信息（可选）
        if args.show_stats:
            print_dataset_statistics(dataset.meta)
        
        # 执行兼容性检查
        result = validate_policy_dataset_compatibility(
            policy_cfg=policy_cfg,
            dataset_meta=dataset.meta,
            rename_map=rename_map,
            raise_on_error=args.raise_on_error,
            verbose=True,
        )
        
        # 返回状态码
        sys.exit(0 if result['passed'] else 1)
        
    except Exception as e:
        logging.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()