"""
SmolVLA 评估脚本 - Piper 7D 机器人

这个脚本封装了评估配置，避免每次都输入长命令行参数。
只需修改下面的配置字典，然后运行：
    python myscripts/eval/eval_smolvla.py

支持的操作：
    - 评估: python myscripts/eval/eval_smolvla.py
    - 验证配置: python myscripts/eval/eval_smolvla.py --validate-only
    - 生成命令: python myscripts/eval/eval_smolvla.py --print-command
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# ============================================================================
# 评估配置 - 在这里修改您的评估参数
# ============================================================================

@dataclass
class EvalConfig:
    """SmolVLA 评估配置
    
    用于在真实机器人上运行策略推理和评估
    """
    
    # ========================================
    # 1. 机器人配置
    # ========================================
    robot_type: str = "piper_follower"
    """机器人类型"""
    
    robot_id: str = "02"
    """机器人 ID（用于识别具体的机器人设备）"""
    
    robot_control_mode: str = "policy"
    """控制模式
    - "policy": 使用策略控制（评估模式）
    - "teleop": 使用遥操作控制（录制模式）
    """
    
    # ========================================
    # 2. 策略配置
    # ========================================
    policy_path: str = "outputs/train/piper_smolvla_transfer_cube_to_bin/checkpoints/last/pretrained_model"
    """策略模型路径
    
    可以是：
    - 本地路径: "outputs/train/xxx/checkpoints/050000/pretrained_model"
    - HuggingFace repo: "username/model_name"
    """
    
    # ========================================
    # 3. 数据集配置（用于保存评估结果）
    # ========================================
    dataset_repo_id: str = "Sprinng/eval_transfer_cube_to_bin"
    """评估数据集 repo ID（用于保存评估结果）"""
    
    dataset_root: str | None = None
    """数据集根目录（默认使用 HuggingFace cache）"""
    
    dataset_single_task: str = "Grab the cube and place it into the bin."
    """任务描述（与训练时保持一致）"""
    
    num_episodes: int = 5
    """评估的 episode 数量"""
    
    episode_time_s: int = 60
    """每个 episode 的最大时长（秒）"""
    
    reset_time_s: int = 30
    """episode 之间的重置时长（秒）"""
    
    fps: int = 30
    """帧率"""
    
    # --- 特征名称映射 ---
    rename_map: dict[str, str] = field(default_factory=lambda: {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
        "observation.images.side": "observation.images.camera3",
    })
    """数据集特征名称到策略特征名称的映射
    
    ⚠️ 重要：必须与训练时使用的 rename_map 一致！
    """
    
    # ========================================
    # 4. 显示和日志配置
    # ========================================
    display_data: bool = True
    """是否显示摄像头画面和数据可视化"""
    
    play_sounds: bool = True
    """是否播放语音提示"""
    
    # ========================================
    # 5. 数据保存配置
    # ========================================
    save_video: bool = True
    """是否保存视频"""
    
    push_to_hub: bool = False
    """评估完成后是否推送到 HuggingFace Hub"""
    
    private: bool = False
    """如果推送到 Hub，是否设为私有"""
    
    tags: list[str] | None = field(default_factory=lambda: ["evaluation", "smolvla", "piper"])
    """数据集标签"""
    
    # ========================================
    # 6. 高级配置
    # ========================================
    resume: bool = False
    """是否从现有数据集恢复评估"""
    
    num_image_writer_processes: int = 0
    """图像写入进程数"""
    
    num_image_writer_threads_per_camera: int = 4
    """每个摄像头的图像写入线程数"""
    
    video_encoding_batch_size: int = 1
    """视频编码批次大小"""


# ============================================================================
# 脚本逻辑
# ============================================================================

def config_to_cli_args(config: EvalConfig) -> list[str]:
    """将配置对象转换为命令行参数列表"""
    args = ["lerobot-record"]
    
    # 机器人配置
    args.extend([
        f"--robot.type={config.robot_type}",
        f"--robot.id={config.robot_id}",
        f"--robot.control_mode={config.robot_control_mode}",
    ])
    
    # 策略配置
    args.append(f"--policy.path={config.policy_path}")
    
    # 数据集配置
    args.extend([
        f"--dataset.repo_id={config.dataset_repo_id}",
        f"--dataset.single_task=\"{config.dataset_single_task}\"",
        f"--dataset.num_episodes={config.num_episodes}",
        f"--dataset.episode_time_s={config.episode_time_s}",
        f"--dataset.reset_time_s={config.reset_time_s}",
        f"--dataset.fps={config.fps}",
        f"--dataset.video={str(config.save_video).lower()}",
        f"--dataset.push_to_hub={str(config.push_to_hub).lower()}",
        # f"--dataset.private={str(config.private).lower()}",
    ])
    
    if config.dataset_root:
        args.append(f"--dataset.root={config.dataset_root}")
    
    # 仅在上传到hub时有用
    # if config.tags:
    #     tags_str = json.dumps(config.tags)
    #     args.append(f"--dataset.tags={tags_str}")
    
    # rename_map（重要！）
    if config.rename_map:
        rename_str = json.dumps(config.rename_map)
        args.append(f"--dataset.rename_map='{rename_str}'")
    
    # 显示和日志配置
    args.extend([
        f"--display_data={str(config.display_data).lower()}",
        f"--play_sounds={str(config.play_sounds).lower()}",
    ])
    
    # 高级配置
    if config.resume:
        args.append("--resume=true")
    
    args.extend([
        f"--dataset.num_image_writer_processes={config.num_image_writer_processes}",
        f"--dataset.num_image_writer_threads_per_camera={config.num_image_writer_threads_per_camera}",
        f"--dataset.video_encoding_batch_size={config.video_encoding_batch_size}",
    ])
    
    return args


def validate_config(config: EvalConfig) -> bool:
    """验证配置的有效性"""
    print("=" * 80)
    print("🔍 Validating Evaluation Configuration")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # 检查策略路径
    policy_path = Path(config.policy_path)
    if not policy_path.exists() and not config.policy_path.startswith(("http://", "https://", "hf://")):
        # 可能是 HuggingFace repo，不检查本地路径
        if "/" not in config.policy_path:
            errors.append(f"Policy path not found and doesn't look like a HuggingFace repo: {config.policy_path}")
    
    # 检查控制模式
    if config.robot_control_mode != "policy":
        warnings.append(
            f"robot_control_mode is '{config.robot_control_mode}', expected 'policy' for evaluation. "
            "Make sure this is intentional."
        )
    
    # 检查 rename_map
    if not config.rename_map:
        warnings.append(
            "rename_map is empty. If your dataset uses different feature names than the policy, "
            "you must provide a rename_map."
        )
    
    # 检查 episode 配置
    if config.num_episodes <= 0:
        errors.append(f"num_episodes must be positive, got {config.num_episodes}")
    
    if config.episode_time_s <= 0:
        errors.append(f"episode_time_s must be positive, got {config.episode_time_s}")
    
    # 检查 Hub 配置
    if config.push_to_hub:
        if "/" not in config.dataset_repo_id:
            errors.append(
                f"dataset_repo_id should be in format 'username/dataset_name', got '{config.dataset_repo_id}'"
            )
    
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


def print_config_summary(config: EvalConfig):
    """打印配置摘要"""
    print("\n" + "=" * 80)
    print("📋 Evaluation Configuration Summary")
    print("=" * 80)
    
    print("\n🤖 Robot:")
    print(f"  Type:                  {config.robot_type}")
    print(f"  ID:                    {config.robot_id}")
    print(f"  Control Mode:          {config.robot_control_mode}")
    
    print("\n🧠 Policy:")
    print(f"  Path:                  {config.policy_path}")
    
    print("\n📊 Dataset (Evaluation Results):")
    print(f"  Repo ID:               {config.dataset_repo_id}")
    print(f"  Task:                  {config.dataset_single_task}")
    print(f"  Num Episodes:          {config.num_episodes}")
    print(f"  Episode Time:          {config.episode_time_s}s")
    print(f"  Reset Time:            {config.reset_time_s}s")
    print(f"  FPS:                   {config.fps}")
    print(f"  Save Video:            {'✅' if config.save_video else '❌'} {config.save_video}")
    
    if config.rename_map:
        print(f"\n🔄 Feature Rename Map:")
        for old_name, new_name in config.rename_map.items():
            print(f"  {old_name}")
            print(f"    → {new_name}")
    
    print("\n📺 Display:")
    print(f"  Display Data:          {'✅' if config.display_data else '❌'} {config.display_data}")
    print(f"  Play Sounds:           {'✅' if config.play_sounds else '❌'} {config.play_sounds}")
    
    print("\n💾 Saving:")
    print(f"  Push to Hub:           {'✅' if config.push_to_hub else '❌'} {config.push_to_hub}")
    if config.push_to_hub:
        print(f"  Private:               {'✅' if config.private else '❌'} {config.private}")
        print(f"  Tags:                  {config.tags}")
    
    if config.resume:
        print("\n♻️  Resume:                ✅ True")
    
    print("\n" + "=" * 80)


def run_evaluation(config: EvalConfig, dry_run: bool = False):
    """运行评估"""
    
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
    print("🚀 Evaluation Command")
    print("=" * 80)
    print("\n" + " \\\n  ".join(cmd_args))
    print("\n" + "=" * 80)
    
    if dry_run:
        print("\n✅ Dry run completed. Command printed above.")
        return
    
    # 确认开始评估
    print("\n⏳ Starting evaluation in 3 seconds... (Ctrl+C to cancel)")
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Evaluation cancelled by user.")
        sys.exit(0)
    
    # 运行评估
    print("\n" + "=" * 80)
    print("🏃 Running Evaluation...")
    print("=" * 80 + "\n")
    
    try:
        cmd_str = " ".join(cmd_args)
        result = subprocess.run(cmd_str, shell=True, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA Evaluation Script for Piper Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 直接开始评估
  python myscripts/eval/eval_smolvla.py
  
  # 只验证配置（不运行评估）
  python myscripts/eval/eval_smolvla.py --validate-only
  
  # 打印命令但不运行
  python myscripts/eval/eval_smolvla.py --print-command
  
  # 使用不同的策略 checkpoint
  python myscripts/eval/eval_smolvla.py --policy-path outputs/train/xxx/checkpoints/050000/pretrained_model
  
  # 评估更多 episodes
  python myscripts/eval/eval_smolvla.py --num-episodes 10
        """
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证配置，不运行评估"
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="打印评估命令但不执行"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="策略模型路径（覆盖配置中的默认值）"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="评估的 episode 数量（覆盖配置中的默认值）"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvalConfig()
    
    # 如果指定了命令行参数，覆盖配置
    if args.policy_path:
        config.policy_path = args.policy_path
    if args.num_episodes:
        config.num_episodes = args.num_episodes
    
    # 根据模式运行
    if args.validate_only:
        print_config_summary(config)
        if validate_config(config):
            print("\n✅ Configuration is valid!")
        else:
            print("\n❌ Configuration has errors!")
            sys.exit(1)
    elif args.print_command:
        run_evaluation(config, dry_run=True)
    else:
        run_evaluation(config, dry_run=False)


if __name__ == "__main__":
    main()