"""
配置验证工具模块

提供策略配置与数据集兼容性检查的工具函数
"""

import logging
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证失败时抛出的异常"""
    pass


def validate_policy_dataset_compatibility(
    policy_cfg: Any,
    dataset_meta: Any,
    rename_map: dict[str, str] | None = None,
    raise_on_error: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    系统性验证策略配置与数据集的兼容性
    
    检查项目：
    1. 特征维度匹配（state, action, images）
    2. 特征类型匹配
    3. 特征名称映射
    4. 归一化配置
    5. 观测步数和动作步数
    
    Args:
        policy_cfg: 策略配置对象
        dataset_meta: 数据集元数据
        rename_map: 特征重命名映射
        raise_on_error: 是否在发现错误时抛出异常
        verbose: 是否打印详细信息
    
    Returns:
        验证结果字典，包含 'passed', 'errors', 'warnings', 'info' 字段
    
    Raises:
        ConfigValidationError: 如果 raise_on_error=True 且发现严重错误
    """
    result = {
        'passed': True,
        'errors': [],      # 严重错误（维度不匹配等）
        'warnings': [],    # 警告信息（缺失特征等）
        'info': [],        # 一般信息
    }
    
    if verbose:
        logger.info("=" * 80)
        logger.info("🔍 Validating policy-dataset compatibility...")
        logger.info("=" * 80)
    
    # 1. 转换数据集特征为策略特征格式
    dataset_features = dataset_to_policy_features(dataset_meta.features)
    
    # 应用 rename_map
    if rename_map:
        if verbose:
            logger.info(f"📝 Applying rename_map: {rename_map}")
        renamed_features = {}
        for key, feature in dataset_features.items():
            new_key = rename_map.get(key, key)
            if new_key != key and verbose:
                logger.info(f"  ✓ Renamed: {key} → {new_key}")
            renamed_features[new_key] = feature
        dataset_features = renamed_features
    
    # 2. 检查输入特征（observations）
    if verbose:
        logger.info("\n📊 Checking INPUT features (observations)...")
        logger.info("-" * 80)
    
    for key, ds_feature in dataset_features.items():
        if ds_feature.type in [FeatureType.STATE, FeatureType.VISUAL]:
            if key in policy_cfg.input_features:
                policy_feature = policy_cfg.input_features[key]
                
                # 检查类型匹配
                if policy_feature.type != ds_feature.type:
                    error_msg = (
                        f"Feature '{key}': type mismatch "
                        f"(policy={policy_feature.type}, dataset={ds_feature.type})"
                    )
                    result['errors'].append(error_msg)
                    if verbose:
                        logger.error(f"  ❌ {error_msg}")
                
                # 检查维度匹配
                if policy_feature.shape != ds_feature.shape:
                    error_msg = (
                        f"Feature '{key}': shape mismatch "
                        f"(policy={policy_feature.shape}, dataset={ds_feature.shape})"
                    )
                    result['errors'].append(error_msg)
                    if verbose:
                        logger.error(f"  ❌ {error_msg}")
                else:
                    if verbose:
                        logger.info(f"  ✓ {key}: {ds_feature.type.name} {ds_feature.shape}")
            else:
                warning_msg = f"Feature '{key}' exists in dataset but missing in policy.input_features"
                result['warnings'].append(warning_msg)
                if verbose:
                    logger.warning(f"  ⚠️  {warning_msg}")
    
    # 检查策略中多余的特征
    for key, policy_feature in policy_cfg.input_features.items():
        if key not in dataset_features:
            warning_msg = f"Feature '{key}' exists in policy but missing in dataset"
            result['warnings'].append(warning_msg)
            if verbose:
                logger.warning(f"  ⚠️  {warning_msg}")
    
    # 3. 检查输出特征（actions）
    if verbose:
        logger.info("\n📤 Checking OUTPUT features (actions)...")
        logger.info("-" * 80)
    
    for key, ds_feature in dataset_features.items():
        if ds_feature.type == FeatureType.ACTION:
            if key in policy_cfg.output_features:
                policy_feature = policy_cfg.output_features[key]
                
                # 检查类型匹配
                if policy_feature.type != ds_feature.type:
                    error_msg = (
                        f"Feature '{key}': type mismatch "
                        f"(policy={policy_feature.type}, dataset={ds_feature.type})"
                    )
                    result['errors'].append(error_msg)
                    if verbose:
                        logger.error(f"  ❌ {error_msg}")
                
                # 检查维度匹配
                if policy_feature.shape != ds_feature.shape:
                    error_msg = (
                        f"Feature '{key}': shape mismatch "
                        f"(policy={policy_feature.shape}, dataset={ds_feature.shape})"
                    )
                    result['errors'].append(error_msg)
                    if verbose:
                        logger.error(f"  ❌ {error_msg}")
                else:
                    if verbose:
                        logger.info(f"  ✓ {key}: {ds_feature.type.name} {ds_feature.shape}")
            else:
                warning_msg = f"Feature '{key}' exists in dataset but missing in policy.output_features"
                result['warnings'].append(warning_msg)
                if verbose:
                    logger.warning(f"  ⚠️  {warning_msg}")
    
    # 4. 检查归一化配置
    if verbose:
        logger.info("\n🔧 Checking normalization configuration...")
        logger.info("-" * 80)
    
    if hasattr(policy_cfg, 'normalization_mapping'):
        if verbose:
            logger.info(f"  Normalization mapping: {policy_cfg.normalization_mapping}")
        
        # 检查每个特征是否有对应的统计数据
        for key in {**policy_cfg.input_features, **policy_cfg.output_features}:
            if key in dataset_meta.stats:
                stats = dataset_meta.stats[key]
                feature = policy_cfg.input_features.get(key, policy_cfg.output_features.get(key))
                norm_mode = policy_cfg.normalization_mapping.get(feature.type, None)
                
                if norm_mode == "MEAN_STD":
                    if "mean" in stats and "std" in stats:
                        if verbose:
                            logger.info(f"  ✓ {key}: has mean/std for MEAN_STD normalization")
                    else:
                        error_msg = f"Feature '{key}': missing mean/std stats for MEAN_STD normalization"
                        result['errors'].append(error_msg)
                        if verbose:
                            logger.error(f"  ❌ {error_msg}")
                elif norm_mode == "MIN_MAX":
                    if "min" in stats and "max" in stats:
                        if verbose:
                            logger.info(f"  ✓ {key}: has min/max for MIN_MAX normalization")
                    else:
                        error_msg = f"Feature '{key}': missing min/max stats for MIN_MAX normalization"
                        result['errors'].append(error_msg)
                        if verbose:
                            logger.error(f"  ❌ {error_msg}")
            else:
                warning_msg = f"Feature '{key}': no stats found in dataset"
                result['warnings'].append(warning_msg)
                if verbose:
                    logger.warning(f"  ⚠️  {warning_msg}")
    
    # 5. 检查观测和动作步数
    if verbose:
        logger.info("\n⏱️  Checking temporal configuration...")
        logger.info("-" * 80)
    
    temporal_config = {}
    for attr in ['n_obs_steps', 'chunk_size', 'n_action_steps']:
        if hasattr(policy_cfg, attr):
            value = getattr(policy_cfg, attr)
            temporal_config[attr] = value
            if verbose:
                logger.info(f"  {attr}: {value}")
    
    result['info'].append(f"Temporal config: {temporal_config}")
    
    # 6. 检查其他重要配置
    if verbose:
        logger.info("\n⚙️  Checking other policy configurations...")
        logger.info("-" * 80)
    
    important_attrs = [
        'device', 'use_amp', 'pretrained_path',
        'max_state_dim', 'max_action_dim',
        'freeze_vision_encoder', 'adapt_to_pi_aloha'
    ]
    
    other_config = {}
    for attr in important_attrs:
        if hasattr(policy_cfg, attr):
            value = getattr(policy_cfg, attr)
            other_config[attr] = value
            if verbose:
                logger.info(f"  {attr}: {value}")
    
    result['info'].append(f"Other config: {other_config}")
    
    # 7. 汇总检查结果
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("📋 Validation Summary")
        logger.info("=" * 80)
    
    if result['errors']:
        result['passed'] = False
        if verbose:
            logger.error(f"❌ Found {len(result['errors'])} critical error(s):")
            for error in result['errors']:
                logger.error(f"  • {error}")
    
    if result['warnings']:
        if verbose:
            logger.warning(f"⚠️  Found {len(result['warnings'])} warning(s):")
            for warning in result['warnings']:
                logger.warning(f"  • {warning}")
    
    if result['passed']:
        if verbose:
            logger.info("✅ All critical checks passed! Policy and dataset are compatible.")
    else:
        if verbose:
            logger.error("❌ Validation FAILED! Please fix the errors above.")
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    # 如果设置了 raise_on_error 且有错误，抛出异常
    if raise_on_error and not result['passed']:
        raise ConfigValidationError(
            f"Policy-dataset compatibility check failed with {len(result['errors'])} error(s). "
            f"See logs above for details."
        )
    
    return result


def print_dataset_statistics(dataset_meta: Any, verbose: bool = True) -> dict[str, Any]:
    """
    打印数据集的详细统计信息
    
    Args:
        dataset_meta: 数据集元数据
        verbose: 是否打印详细信息
    
    Returns:
        统计信息字典
    """
    stats_summary = {}
    
    if verbose:
        logger.info("=" * 80)
        logger.info("📈 Dataset Statistics")
        logger.info("=" * 80)
    
    for key, stats in dataset_meta.stats.items():
        stats_summary[key] = {}
        if verbose:
            logger.info(f"\n{key}:")
        
        for stat_name, stat_value in stats.items():
            if isinstance(stat_value, (list, np.ndarray)):
                stat_array = np.array(stat_value)
                stats_summary[key][stat_name] = {
                    'shape': stat_array.shape,
                    'dtype': str(stat_array.dtype),
                }
                if verbose:
                    logger.info(f"  {stat_name}: shape={stat_array.shape}, dtype={stat_array.dtype}")
                    if stat_array.size <= 10:
                        logger.info(f"    values={stat_value}")
            else:
                stats_summary[key][stat_name] = stat_value
                if verbose:
                    logger.info(f"  {stat_name}: {stat_value}")
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    return stats_summary


def compare_policy_configs(
    config1: Any,
    config2: Any,
    config1_name: str = "Config 1",
    config2_name: str = "Config 2",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    比较两个策略配置的差异
    
    Args:
        config1: 第一个配置对象
        config2: 第二个配置对象
        config1_name: 第一个配置的名称
        config2_name: 第二个配置的名称
        verbose: 是否打印详细信息
    
    Returns:
        差异信息字典
    """
    differences = {
        'input_features': {},
        'output_features': {},
        'other_attributes': {},
    }
    
    if verbose:
        logger.info("=" * 80)
        logger.info("🔄 Comparing Policy Configurations")
        logger.info("=" * 80)
        logger.info(f"{config1_name} vs {config2_name}\n")
    
    # 比较 input_features
    if verbose:
        logger.info("📊 Input Features:")
    
    all_input_keys = set(config1.input_features.keys()) | set(config2.input_features.keys())
    
    for key in sorted(all_input_keys):
        feat1 = config1.input_features.get(key)
        feat2 = config2.input_features.get(key)
        
        if feat1 is None:
            differences['input_features'][key] = f"Only in {config2_name}"
            if verbose:
                logger.warning(f"  {key}: Only in {config2_name}")
        elif feat2 is None:
            differences['input_features'][key] = f"Only in {config1_name}"
            if verbose:
                logger.warning(f"  {key}: Only in {config1_name}")
        elif feat1.shape != feat2.shape or feat1.type != feat2.type:
            diff_info = (
                f"{config1_name}: {feat1.type}[{feat1.shape}] vs "
                f"{config2_name}: {feat2.type}[{feat2.shape}]"
            )
            differences['input_features'][key] = diff_info
            if verbose:
                logger.warning(f"  {key}: {diff_info}")
        else:
            if verbose:
                logger.info(f"  {key}: ✓ Same ({feat1.type}[{feat1.shape}])")
    
    # 比较 output_features
    if verbose:
        logger.info("\n📤 Output Features:")
    
    all_output_keys = set(config1.output_features.keys()) | set(config2.output_features.keys())
    
    for key in sorted(all_output_keys):
        feat1 = config1.output_features.get(key)
        feat2 = config2.output_features.get(key)
        
        if feat1 is None:
            differences['output_features'][key] = f"Only in {config2_name}"
            if verbose:
                logger.warning(f"  {key}: Only in {config2_name}")
        elif feat2 is None:
            differences['output_features'][key] = f"Only in {config1_name}"
            if verbose:
                logger.warning(f"  {key}: Only in {config1_name}")
        elif feat1.shape != feat2.shape or feat1.type != feat2.type:
            diff_info = (
                f"{config1_name}: {feat1.type}[{feat1.shape}] vs "
                f"{config2_name}: {feat2.type}[{feat2.shape}]"
            )
            differences['output_features'][key] = diff_info
            if verbose:
                logger.warning(f"  {key}: {diff_info}")
        else:
            if verbose:
                logger.info(f"  {key}: ✓ Same ({feat1.type}[{feat1.shape}])")
    
    # 比较其他重要属性
    if verbose:
        logger.info("\n⚙️  Other Attributes:")
    
    important_attrs = [
        'n_obs_steps', 'chunk_size', 'n_action_steps',
        'device', 'use_amp', 'normalization_mapping',
        'max_state_dim', 'max_action_dim',
    ]
    
    for attr in important_attrs:
        val1 = getattr(config1, attr, None)
        val2 = getattr(config2, attr, None)
        
        if val1 != val2:
            diff_info = f"{config1_name}: {val1} vs {config2_name}: {val2}"
            differences['other_attributes'][attr] = diff_info
            if verbose:
                logger.warning(f"  {attr}: {diff_info}")
        else:
            if verbose:
                logger.info(f"  {attr}: ✓ Same ({val1})")
    
    if verbose:
        logger.info("=" * 80 + "\n")
    
    return differences