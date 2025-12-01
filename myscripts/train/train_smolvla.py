"""
SmolVLA 训练脚本 - Piper 7D 机器人

这个脚本封装了训练配置，避免每次都输入长命令行参数。
只需修改下面的配置字典，然后运行：
    python myscripts/train/train_smolvla.py

支持的操作：
    - 训练: python myscripts/train/train_smolvla.py
    - 验证配置: python myscripts/train/train_smolvla.py --validate-only
    - 生成命令: python myscripts/train/train_smolvla.py --print-command
    - 恢复训练: python myscripts/train/train_smolvla.py --resume <checkpoint_path>

    export CUDA_VISIBLE_DEVICES=5
    - 恢复训练: python myscripts/train/train_smolvla.py --resume
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# ============================================================================
# 训练配置 - 在这里修改您的训练参数
# ============================================================================

@dataclass
class TrainingConfig:
    """SmolVLA 训练配置
    
    参数参考:
    - configuration_smolvla.py: SmolVLA 策略的默认配置
    - lerobot/smolvla_base: 预训练模型的配置
    """
    
    # ========================================
    # 1. 策略配置
    # ========================================
    policy_path: str = "lerobot/smolvla_base"
    """预训练模型路径或 HuggingFace repo ID"""
    
    # --- VLM 权重加载 ---
    load_vlm_weights: bool = True
    """是否加载预训练的 VLM 权重
    
    """
    
    # --- 训练策略 ---
    freeze_vision_encoder: bool = True
    """是否冻结视觉编码器
    - True: 只训练 Action Expert（快速微调，推荐）
    - False: 也训练视觉编码器（需要更多数据，避免过拟合）
    
    默认值: True（来自 configuration_smolvla.py）
    """
    
    train_expert_only: bool = True
    """是否只训练 Action Expert 层
    - True: 只训练 expert 层（快速，推荐）
    - False: 训练整个模型（慢，需要更多数据）
    
    默认值: True（来自 configuration_smolvla.py）
    """
    
    train_state_proj: bool = True
    """是否训练 state projection 层
    
    默认值: True（来自 configuration_smolvla.py）
    """
    
    # --- 图像预处理 ---
    resize_imgs_with_padding: tuple[int, int] | None = (512, 512)
    """图像 resize 尺寸 (width, height)
    - (512, 512): 标准尺寸（来自 configuration_smolvla.py）
    - None: 使用原始分辨率（不推荐，需要更多显存）
    
    注意: SmolVLA 会自动保持宽高比并 padding
    """
    
    # --- ALOHA 专用配置（Piper 不需要） ---
    adapt_to_pi_aloha: bool = False
    """是否适配 Physical Intelligence 的 ALOHA 空间
    - True: 用于 ALOHA 机器人
    - False: 用于其他机器人（如 Piper）
    
    默认值: True（configuration_smolvla.py），但 Piper 应该设为 False
    """
    
    use_delta_joint_actions_aloha: bool = False
    """是否使用关节增量动作（ALOHA 专用）
    
    注意: 目前未在 LeRobot 中实现，保持 False
    """
    
    # ========================================
    # 2. 数据集配置
    # ========================================
    dataset_repo_id: str = "Sprinng/piper_transfer_cube_to_bin"
    """数据集 HuggingFace repo ID"""
    
    # 特征名称映射（数据集名称 → 策略名称）
    rename_map: dict[str, str] = field(default_factory=lambda: {
        "observation.images.top_rgb": "observation.images.camera1",
        "observation.images.wrist_rgb": "observation.images.camera2",
        "observation.images.side_rgb": "observation.images.camera3",
    })
    """数据集特征名称到策略特征名称的映射"""
    
    # ========================================
    # 3. 模型架构配置
    # ========================================
    state_dim: int = 7
    """状态维度（Piper: 6关节 + 1夹爪 = 7）"""
    
    action_dim: int = 7
    """动作维度（Piper: 6关节 + 1夹爪 = 7）"""
    
    max_state_dim: int = 32
    """最大状态维度（用于 padding）
    
    默认值: 32（来自 configuration_smolvla.py）
    """
    
    max_action_dim: int = 32
    """最大动作维度（用于 padding）
    
    默认值: 32（来自 configuration_smolvla.py）
    """
    
    # --- 时间配置 ---
    n_obs_steps: int = 1
    """观察步数（历史帧数）
    
    默认值: 1（来自 configuration_smolvla.py）
    """
    
    chunk_size: int = 50
    """预测的动作序列长度（action chunk）
    
    默认值: 50（来自 configuration_smolvla.py）
    """
    
    n_action_steps: int = 50
    """执行的动作步数（每次推理执行多少步）
    
    默认值: 50（来自 configuration_smolvla.py）
    注意: n_action_steps <= chunk_size
    """
    
    # ========================================
    # 4. 训练超参数
    # ========================================
    batch_size: int = 128
    """训练批次大小
    
    推荐值:
    - 32: 适合 24GB GPU（freeze_vision_encoder=True）
    - 16: 适合 16GB GPU 或 freeze_vision_encoder=False
    - 8: 显存不足时
    """
    
    training_steps: int = 50000
    """训练总步数
    
    推荐值:
    - 30000-50000: 快速微调（freeze_vision_encoder=True）
    - 100000-200000: 深度微调（freeze_vision_encoder=False）
    - 500000+: 从头训练（load_vlm_weights=False）
    """
    
    # --- 优化器配置 ---
    learning_rate: float | None = None
    """学习率
    - None: 使用策略默认值（1e-4，来自 configuration_smolvla.py）
    - 自定义值: 例如 5e-5（用于深度微调）
    """
    
    optimizer_betas: tuple[float, float] | None = None
    """AdamW 优化器的 beta 参数
    
    默认值: (0.9, 0.95)（来自 configuration_smolvla.py）
    """
    
    optimizer_eps: float | None = None
    """AdamW 优化器的 epsilon 参数
    
    默认值: 1e-8（来自 configuration_smolvla.py）
    """
    
    optimizer_weight_decay: float | None = None
    """权重衰减
    
    默认值: 1e-10（来自 configuration_smolvla.py）
    """
    
    grad_clip_norm: float | None = None
    """梯度裁剪范数
    
    默认值: 10（来自 configuration_smolvla.py）
    """
    
    # --- 学习率调度器配置 ---
    scheduler_warmup_steps: int | None = None
    """学习率预热步数
    
    默认值: 1000（来自 configuration_smolvla.py）
    """
    
    scheduler_decay_steps: int | None = None
    """学习率衰减步数
    
    默认值: 30000（来自 configuration_smolvla.py）
    """
    
    scheduler_decay_lr: float | None = None
    """学习率衰减到的最终值
    
    默认值: 2.5e-6（来自 configuration_smolvla.py）
    """
    
    # --- 评估和保存 ---
    eval_freq: int = 10000
    """评估频率（每多少步评估一次）"""
    
    save_freq: int = 10000
    """保存频率（每多少步保存一次 checkpoint）"""
    
    # ========================================
    # 5. 输出配置
    # ========================================
    output_dir: str = "outputs/train/piper_smolvla_transfer_cube_to_bin_128"
    """训练输出目录"""
    
    job_name: str = "smolvla_transfer_cube_to_bin_128"
    """任务名称（用于日志和 wandb）"""
    
    # ========================================
    # 6. 日志配置
    # ========================================
    use_wandb: bool = True
    """是否使用 Weights & Biases 记录训练"""
    
    wandb_project: str | None = "lerobot"
    """W&B 项目名称（如果 use_wandb=True）"""
    
    wandb_entity: str | None = None
    """W&B 实体（团队或用户名）"""
    
    wandb_disable_artifact: bool = True  # ← 新增参数
    """是否禁用 W&B Artifact 功能
    - True: 禁用 Artifact（推荐，减少存储和上传开销）
    - False: 启用 Artifact（会自动保存模型和数据集版本）
    
    默认值: True（推荐）
    """

    log_freq: int = 100
    """日志记录频率"""
    
    # ========================================
    # 7. 其他配置
    # ========================================
    device: str = "cuda"
    """训练设备（cuda, cpu, mps）"""
    
    num_workers: int = 4
    """数据加载器的工作进程数"""
    
    seed: int = 1000
    """随机种子"""
    
    resume: bool = False
    """是否从 checkpoint 恢复训练
    
    - True: 从 output_dir/checkpoints/last 恢复训练
    - False: 从头开始训练
    
    注意: checkpoint 路径由 output_dir 自动推导，无需手动指定
    """
    
    push_to_hub: bool = False
    """训练完成后是否推送到 HuggingFace Hub"""
    
    hub_repo_id: str | None = None
    """HuggingFace Hub 仓库 ID（如果 push_to_hub=True）"""
    
    # ========================================
    # 8. 高级配置（通常不需要修改）
    # ========================================
    vlm_model_name: str | None = None
    """VLM 模型名称
    
    默认值: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    （来自 configuration_smolvla.py）
    
    注意: 使用 from_pretrained 时会自动加载，通常不需要手动指定
    """
    
    tokenizer_max_length: int | None = None
    """Tokenizer 最大长度
    
    默认值: 48（来自 configuration_smolvla.py）
    """
    
    num_steps: int | None = None
    """解码步数
    
    默认值: 10（来自 configuration_smolvla.py）
    """


# ============================================================================
# 训练脚本逻辑
# ============================================================================

def config_to_cli_args(config: TrainingConfig) -> list[str]:
    """将配置对象转换为命令行参数列表"""
    args = ["lerobot-train"]
    
    # 策略基本配置
    args.extend([
        f"--policy.path={config.policy_path}",
    ])
    
    # VLM 权重和训练策略
    args.extend([
        f"--policy.load_vlm_weights={str(config.load_vlm_weights).lower()}",
        f"--policy.freeze_vision_encoder={str(config.freeze_vision_encoder).lower()}",
        f"--policy.train_expert_only={str(config.train_expert_only).lower()}",
        f"--policy.train_state_proj={str(config.train_state_proj).lower()}",
    ])
    
    # ALOHA 适配（Piper 应该为 False）
    if not config.adapt_to_pi_aloha:
        args.append(f"--policy.adapt_to_pi_aloha=false")
    
    # 图像预处理
    if config.resize_imgs_with_padding:
        w, h = config.resize_imgs_with_padding
        args.append(f"--policy.resize_imgs_with_padding=[{w},{h}]")
    
    # ⭐⭐⭐ 修正：使用完整的 JSON 对象来指定特征维度 ⭐⭐⭐
    # 构建 input_features 对象
    input_features = {
        "observation.state": {
            "type": "STATE",
            "shape": [config.state_dim]
        }
    }
    # 转换为 JSON 字符串并转义引号
    input_features_json = json.dumps(input_features).replace('"', '\\"')
    args.append(f'--policy.input_features="{input_features_json}"')
    
    # 构建 output_features 对象
    output_features = {
        "action": {
            "type": "ACTION",
            "shape": [config.action_dim]
        }
    }
    # 转换为 JSON 字符串并转义引号
    output_features_json = json.dumps(output_features).replace('"', '\\"')
    args.append(f'--policy.output_features="{output_features_json}"')
    
    # 模型架构配置（如果与默认值不同）
    if config.max_state_dim != 32:
        args.append(f"--policy.max_state_dim={config.max_state_dim}")
    if config.max_action_dim != 32:
        args.append(f"--policy.max_action_dim={config.max_action_dim}")
    
    # 时间配置（如果与默认值不同）
    if config.n_obs_steps != 1:
        args.append(f"--policy.n_obs_steps={config.n_obs_steps}")
    if config.chunk_size != 50:
        args.append(f"--policy.chunk_size={config.chunk_size}")
    if config.n_action_steps != 50:
        args.append(f"--policy.n_action_steps={config.n_action_steps}")
    
    # 数据集配置
    args.append(f"--dataset.repo_id={config.dataset_repo_id}")
    
    if config.rename_map:
        # ⭐ 使用单引号包裹 JSON，内部使用双引号
        rename_str = json.dumps(config.rename_map)
        args.append(f"--rename_map='{rename_str}'")
    
    # 训练超参数
    args.extend([
        f"--batch_size={config.batch_size}",
        f"--steps={config.training_steps}",
    ])
    
    # 优化器配置（只覆盖非默认值）
    if config.learning_rate is not None:
        args.append(f"--policy.optimizer_lr={config.learning_rate}")
    if config.optimizer_betas is not None:
        args.append(f"--policy.optimizer_betas={list(config.optimizer_betas)}")
    if config.optimizer_eps is not None:
        args.append(f"--policy.optimizer_eps={config.optimizer_eps}")
    if config.optimizer_weight_decay is not None:
        args.append(f"--policy.optimizer_weight_decay={config.optimizer_weight_decay}")
    if config.grad_clip_norm is not None:
        args.append(f"--policy.optimizer_grad_clip_norm={config.grad_clip_norm}")
    
    # 学习率调度器配置
    if config.scheduler_warmup_steps is not None:
        args.append(f"--policy.scheduler_warmup_steps={config.scheduler_warmup_steps}")
    if config.scheduler_decay_steps is not None:
        args.append(f"--policy.scheduler_decay_steps={config.scheduler_decay_steps}")
    if config.scheduler_decay_lr is not None:
        args.append(f"--policy.scheduler_decay_lr={config.scheduler_decay_lr}")
    
    # 评估和保存
    args.extend([
        f"--eval_freq={config.eval_freq}",
        f"--save_freq={config.save_freq}",
    ])
    
    # 输出配置
    args.extend([
        f"--output_dir={config.output_dir}",
        f"--job_name={config.job_name}",
    ])
    
    # 日志配置
    args.append(f"--wandb.enable={str(config.use_wandb).lower()}")
    
    if config.use_wandb:
        if config.wandb_project:
            args.append(f"--wandb.project={config.wandb_project}")
        if config.wandb_entity:
            args.append(f"--wandb.entity={config.wandb_entity}")
        
        # 禁用 W&B Artifact
        if config.wandb_disable_artifact:
            args.append(f"--wandb.disable_artifact={str(config.wandb_disable_artifact).lower()}")
    
    args.append(f"--log_freq={config.log_freq}")
    
    # 其他配置
    args.extend([
        f"--policy.device={config.device}",
        f"--num_workers={config.num_workers}",
        f"--seed={config.seed}",
        f"--policy.push_to_hub={str(config.push_to_hub).lower()}",
    ])
    
    if config.push_to_hub and config.hub_repo_id:
        args.append(f"--policy.repo_id={config.hub_repo_id}")
    
    # ⭐ 修改：resume 是布尔值，不是路径
    if config.resume:
        args.append("--resume=true")
    
    return args


def validate_config(config: TrainingConfig) -> bool:
    """验证配置的有效性"""
    print("=" * 80)
    print("🔍 Validating Configuration")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # 检查维度
    if config.state_dim <= 0 or config.action_dim <= 0:
        errors.append(f"Invalid dimensions: state_dim={config.state_dim}, action_dim={config.action_dim}")
    
    if config.state_dim > config.max_state_dim:
        errors.append(f"state_dim ({config.state_dim}) > max_state_dim ({config.max_state_dim})")
    
    if config.action_dim > config.max_action_dim:
        errors.append(f"action_dim ({config.action_dim}) > max_action_dim ({config.max_action_dim})")
    
    # 检查时间配置
    if config.n_action_steps > config.chunk_size:
        errors.append(f"n_action_steps ({config.n_action_steps}) > chunk_size ({config.chunk_size})")
    
    # 检查 ALOHA 配置
    if config.adapt_to_pi_aloha and "piper" in config.dataset_repo_id.lower():
        warnings.append(
            "adapt_to_pi_aloha=True but using Piper dataset. "
            "This setting is for Physical Intelligence ALOHA robots. "
            "Consider setting it to False for Piper."
        )
    
    if config.use_delta_joint_actions_aloha:
        errors.append("use_delta_joint_actions_aloha is not implemented yet in LeRobot")
    
    # 检查训练策略
    if not config.load_vlm_weights:
        warnings.append(
            "load_vlm_weights=False: Training from scratch. "
            "This requires large amounts of data and training time. "
            "Consider setting it to True for finetuning."
        )
    
    if config.load_vlm_weights and not config.freeze_vision_encoder and config.batch_size > 16:
        warnings.append(
            f"Training vision encoder with batch_size={config.batch_size} may require too much GPU memory. "
            "Consider reducing batch_size to 8-16."
        )
    
    # 检查 resume 配置
    if config.resume:
        # 检查 last checkpoint 是否存在
        checkpoint_path = Path(config.output_dir) / "checkpoints" / "last"
        if not checkpoint_path.exists():
            errors.append(
                f"Resume is enabled but checkpoint not found: {checkpoint_path}\n"
                f"  Make sure the output directory contains a valid checkpoint with 'last' symlink."
            )
    
    # 检查输出目录
    output_path = Path(config.output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not config.resume:
        warnings.append(
            f"Output directory already exists and is not empty: {output_path}\n"
            f"  Consider using --resume to continue training, or change output_dir."
        )
    
    # 检查 Hub 配置
    if config.push_to_hub and not config.hub_repo_id:
        errors.append("push_to_hub=True but hub_repo_id is not set")
    
    # 检查 W&B 配置
    if config.use_wandb and not config.wandb_project:
        warnings.append("wandb.enable=True but wandb_project is not set (will use default)")
    
    
    # 打印结果
    if errors:
        print("\n❌ Validation FAILED:")
        for err in errors:
            print(f"  • {err}")
        print()
        return False
    
    if warnings:
        print("\n⚠️  Warnings:")
        for warn in warnings:
            print(f"  • {warn}")
    
    print("\n✅ Configuration validation passed!")
    print("=" * 80)
    return True


def print_config_summary(config: TrainingConfig):
    """打印配置摘要"""
    print("\n" + "=" * 80)
    print("📋 Training Configuration Summary")
    print("=" * 80)
    
    print("\n🤖 Policy:")
    print(f"  Path:                  {config.policy_path}")
    print(f"  VLM Model:             {config.vlm_model_name or 'default'}")
    print(f"  Load VLM Weights:      {'✅' if config.load_vlm_weights else '❌'} {config.load_vlm_weights}")
    print(f"  Freeze Vision:         {'✅' if config.freeze_vision_encoder else '❌'} {config.freeze_vision_encoder}")
    print(f"  Train Expert Only:     {'✅' if config.train_expert_only else '❌'} {config.train_expert_only}")
    print(f"  Train State Proj:      {'✅' if config.train_state_proj else '❌'} {config.train_state_proj}")
    print(f"  Image Resize:          {config.resize_imgs_with_padding}")
    print(f"  Adapt to Pi ALOHA:     {'✅' if config.adapt_to_pi_aloha else '❌'} {config.adapt_to_pi_aloha}")
    
    print("\n📊 Dataset:")
    print(f"  Repo ID:               {config.dataset_repo_id}")
    print(f"  Rename Map:            {len(config.rename_map)} mappings")
    
    print("\n🔢 Dimensions:")
    print(f"  State Dimension:       {config.state_dim} (max: {config.max_state_dim})")
    print(f"  Action Dimension:      {config.action_dim} (max: {config.max_action_dim})")
    
    print("\n⏱️  Temporal:")
    print(f"  Observation Steps:     {config.n_obs_steps}")
    print(f"  Chunk Size:            {config.chunk_size}")
    print(f"  Action Steps:          {config.n_action_steps}")
    
    print("\n⚙️  Training:")
    print(f"  Batch Size:            {config.batch_size}")
    print(f"  Training Steps:        {config.training_steps:,}")
    print(f"  Learning Rate:         {config.learning_rate or 'default (1e-4)'}")
    print(f"  Warmup Steps:          {config.scheduler_warmup_steps or 'default (1000)'}")
    print(f"  Decay Steps:           {config.scheduler_decay_steps or 'default (30000)'}")
    print(f"  Device:                {config.device}")
    
    print("\n💾 Checkpointing:")
    print(f"  Eval Frequency:        every {config.eval_freq} steps")
    print(f"  Save Frequency:        every {config.save_freq} steps")
    if config.resume:
        checkpoint_path = Path(config.output_dir) / "checkpoints" / "last"
        print(f"  Resume from:           {checkpoint_path}")
    
    print("\n📂 Output:")
    print(f"  Output Directory:      {config.output_dir}")
    print(f"  Job Name:              {config.job_name}")
    
    print("\n📈 Logging:")
    print(f"  Use W&B:               {'✅' if config.use_wandb else '❌'} {config.use_wandb}")
    if config.use_wandb:
        print(f"  W&B Project:           {config.wandb_project or 'default'}")
    print(f"  Log Frequency:         every {config.log_freq} steps")
    
    if config.push_to_hub:
        print(f"\n🤗 HuggingFace Hub:")
        print(f"  Push to Hub:           ✅ True")
        print(f"  Hub Repo ID:           {config.hub_repo_id}")
    
    print("\n" + "=" * 80)


def run_training(config: TrainingConfig, dry_run: bool = False):
    """运行训练"""
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 验证配置
    if not validate_config(config):
        print("\n❌ Please fix the configuration errors above.")
        sys.exit(1)
    
    # 生成命令
    cmd_args = config_to_cli_args(config)
    
    # 打印命令
    print("\n" + "=" * 80)
    print("🚀 Training Command")
    print("=" * 80)
    print("\n" + " \\\n  ".join(cmd_args))
    print("\n" + "=" * 80)
    
    if dry_run:
        print("\n✅ Dry run completed. Command printed above.")
        return
    
    # 确认开始训练
    print("\n⏳ Starting training in 3 seconds... (Ctrl+C to cancel)")
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Training cancelled by user.")
        sys.exit(0)
    
    # 运行训练
    print("\n" + "=" * 80)
    print("🏃 Running Training...")
    print("=" * 80 + "\n")
    
    try:
        # ⭐ 使用 shell=True 来运行 CLI 命令
        cmd_str = " ".join(cmd_args)
        result = subprocess.run(cmd_str, shell=True, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA Training Script for Piper Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 直接开始训练
  python myscripts/train/train_smolvla.py
  
  # 只验证配置（不运行训练）
  python myscripts/train/train_smolvla.py --validate-only
  
  # 打印命令但不运行
  python myscripts/train/train_smolvla.py --print-command
  
  # 从 checkpoint 恢复训练（自动使用 output_dir/checkpoints/last）
  python myscripts/train/train_smolvla.py --resume
  
  # 从不同的训练目录恢复
  python myscripts/train/train_smolvla.py --resume --output-dir outputs/train/another_experiment
        """
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证配置，不运行训练"
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="打印训练命令但不执行"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint 恢复训练（自动使用 output_dir/checkpoints/last）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="覆盖配置中的 output_dir（用于从不同目录恢复训练）"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig()
    
    # 如果指定了 resume，启用恢复模式
    if args.resume:
        config.resume = True
    
    # 如果指定了 output_dir，覆盖配置
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # 根据模式运行
    if args.validate_only:
        print_config_summary(config)
        if validate_config(config):
            print("\n✅ Configuration is valid!")
        else:
            print("\n❌ Configuration has errors!")
            sys.exit(1)
    elif args.print_command:
        run_training(config, dry_run=True)
    else:
        run_training(config, dry_run=False)


if __name__ == "__main__":
    main()