"""
SmolVLA 泛化性能评估实验配置

实验设计：5大维度 × 多个难度级别
- 维度 1：物体属性泛化（颜色、形状、尺寸、材质、重量）
- 维度 2：目标位置泛化（位置、高度、朝向、距离、容器位置）
- 维度 3：环境干扰泛化（干扰物、背景、光照、遮挡、动态）
- 维度 4：操作模式泛化（直接、推动、绕行、堆叠、双手）
- 维度 5：语言指令泛化（原始、改写、描述词、空间、复杂）

使用方法：
    # 运行单个实验
    python myscripts/eval/eval_smolvla_generalization.py --experiment 1.1
    
    # 运行整个阶段的实验
    python myscripts/eval/eval_smolvla_generalization.py --phase 1
    
    # 运行所有实验
    python myscripts/eval/eval_smolvla_generalization.py --all
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """单个泛化实验的配置"""
    
    # 基本信息
    id: str
    """实验 ID（例如：1.1, 2.3）"""
    
    name: str
    """实验名称"""
    
    dimension: Literal["object", "spatial", "environment", "manipulation", "language"]
    """泛化维度"""
    
    difficulty: Literal["easy", "medium", "hard"]
    """难度级别"""
    
    priority: Literal["P0", "P1", "P2", "P3"]
    """优先级（P0=必做，P1=推荐，P2=可选，P3=研究）"""
    
    description: str
    """实验描述"""
    
    # 实验参数
    task_description: str
    """任务描述（传递给策略的语言指令）"""
    
    num_episodes: int = 10
    """评估的 episode 数量"""
    
    dataset_suffix: str = ""
    """数据集名称后缀（例如：_red_cube, _blue_cube）"""
    
    # 实验特定配置（用于记录实验设置）
    modifications: dict[str, str] = field(default_factory=dict)
    """实验中的物理修改（例如：{"cube_color": "blue", "bin_position": "left"}）"""
    
    expected_metrics: dict[str, float] = field(default_factory=dict)
    """预期的性能指标（例如：{"success_rate": 0.8}）"""


# ============================================================================
# 实验库：5大维度 × 多个实验
# ============================================================================

EXPERIMENTS: dict[str, ExperimentConfig] = {
    # ========================================================================
    # 维度 1：物体属性泛化
    # ========================================================================
    "1.1": ExperimentConfig(
        id="1.1",
        name="改变目标颜色",
        dimension="object",
        difficulty="easy",
        priority="P0",
        description="将橙色方块改为蓝色/绿色/黄色方块，测试视觉特征泛化能力",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_blue_cube",
        modifications={
            "cube_color": "blue",  # 或 "green", "yellow"
        },
        expected_metrics={
            "success_rate": 0.85,  # 预期成功率
            "grasp_accuracy": 0.90,  # 预期抓取准确度
        },
    ),
    
    "1.2": ExperimentConfig(
        id="1.2",
        name="改变物体形状",
        dimension="object",
        difficulty="medium",
        priority="P1",
        description="将方块改为圆柱/球体/长方体，测试抓取策略的通用性",
        task_description="Grab the orange cuboid and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_cuboid",
        modifications={
            "object_shape": "cylinder",  # 或 "sphere", "cuboid"
        },
        expected_metrics={
            "success_rate": 0.70,
            "grasp_stability": 0.75,
        },
    ),
    
    "1.3": ExperimentConfig(
        id="1.3",
        name="改变物体尺寸",
        dimension="object",
        difficulty="medium",
        priority="P2",
        description="将标准方块改为迷你/巨大方块，测试夹爪适应性",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_large_cube",
        modifications={
            "cube_size": "large",  # 或 "mini"
        },
        expected_metrics={
            "success_rate": 0.65,
            "gripper_adaptation": 0.70,
        },
    ),
    
    # ========================================================================
    # 维度 2：目标位置泛化
    # ========================================================================
    "2.1": ExperimentConfig(
        id="2.1",
        name="推动目标物体",
        dimension="spatial",
        difficulty="easy",
        priority="P0",
        description="方块位置随机偏移 ±5cm，测试空间泛化能力",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=15,
        dataset_suffix="_shifted_position",
        modifications={
            "cube_position": "random_shift_5cm",
        },
        expected_metrics={
            "success_rate": 0.80,
            "trajectory_smoothness": 0.85,
        },
    ),
    
    "2.3": ExperimentConfig(
        id="2.3",
        name="改变目标朝向",
        dimension="spatial",
        difficulty="medium",
        priority="P1",
        description="方块旋转 0°/45°/90°，测试抓取角度适应性",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_rotated_45deg",
        modifications={
            "cube_rotation": "45_degrees",  # 或 "90_degrees"
        },
        expected_metrics={
            "success_rate": 0.75,
            "grasp_angle_adaptation": 0.80,
        },
    ),
    
    "2.5": ExperimentConfig(
        id="2.5",
        name="移动目标容器",
        dimension="spatial",
        difficulty="hard",
        priority="P2",
        description="Bin 位置/朝向随机化，测试放置阶段的泛化能力",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_moved_bin",
        modifications={
            "bin_position": "random_shift_10cm",
            "bin_orientation": "rotated_30deg",
        },
        expected_metrics={
            "success_rate": 0.60,
            "placement_accuracy": 0.65,
        },
    ),
    
    # ========================================================================
    # 维度 3：环境干扰泛化
    # ========================================================================
    "3.1": ExperimentConfig(
        id="3.1",
        name="增加其他物体",
        dimension="environment",
        difficulty="medium",
        priority="P0",
        description="添加 1-3 个干扰物体（不同颜色），测试目标识别和抗干扰能力",
        task_description="Grab the red cube and place it into the bin.",
        num_episodes=15,
        dataset_suffix="_with_distractors",
        modifications={
            "distractor_count": "3",
            "distractor_colors": "blue, green, yellow",
        },
        expected_metrics={
            "success_rate": 0.70,
            "target_recognition_accuracy": 0.85,
        },
    ),
    
    "3.2": ExperimentConfig(
        id="3.2",
        name="改变背景",
        dimension="environment",
        difficulty="easy",
        priority="P1",
        description="不同颜色/纹理的桌面，测试视觉鲁棒性",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_dark_background",
        modifications={
            "background": "dark",  # 或 "textured", "patterned"
        },
        expected_metrics={
            "success_rate": 0.85,
            "visual_robustness": 0.90,
        },
    ),
    
    "3.3": ExperimentConfig(
        id="3.3",
        name="改变光照条件",
        dimension="environment",
        difficulty="medium",
        priority="P1",
        description="明亮/昏暗/侧光/顶光条件下，测试图像质量影响",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_low_light",
        modifications={
            "lighting": "low_light",  # 或 "bright", "side_light"
        },
        expected_metrics={
            "success_rate": 0.75,
            "image_quality_impact": 0.80,
        },
    ),
    
    "3.4": ExperimentConfig(
        id="3.4",
        name="添加遮挡物",
        dimension="environment",
        difficulty="hard",
        priority="P2",
        description="部分遮挡目标物体，测试推理能力",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_partial_occlusion",
        modifications={
            "occlusion": "partial",  # 遮挡 30-50% 的目标物体
        },
        expected_metrics={
            "success_rate": 0.55,
            "reasoning_ability": 0.60,
        },
    ),
    
    # ========================================================================
    # 维度 4：操作模式泛化
    # ========================================================================
    "4.1": ExperimentConfig(
        id="4.1",
        name="基线：直接抓取放置",
        dimension="manipulation",
        difficulty="easy",
        priority="P0",
        description="原始任务作为基线，用于对比其他实验的性能",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=20,
        dataset_suffix="_baseline",
        modifications={},
        expected_metrics={
            "success_rate": 0.90,  # 基线性能
        },
    ),
    
    "4.2": ExperimentConfig(
        id="4.2",
        name="推动后抓取",
        dimension="manipulation",
        difficulty="medium",
        priority="P2",
        description="先推动方块到目标区域，再抓取，测试两阶段控制能力",
        task_description="Push the cube to the target area, then grab it and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_push_then_grasp",
        modifications={
            "manipulation_mode": "push_then_grasp",
        },
        expected_metrics={
            "success_rate": 0.60,
            "two_stage_control": 0.65,
        },
    ),
    
    "4.3": ExperimentConfig(
        id="4.3",
        name="绕过障碍物",
        dimension="manipulation",
        difficulty="hard",
        priority="P3",
        description="Bin 周围放置障碍物，测试路径规划能力",
        task_description="Grab the cube and place it into the bin, avoiding obstacles.",
        num_episodes=10,
        dataset_suffix="_with_obstacles",
        modifications={
            "obstacles": "around_bin",
        },
        expected_metrics={
            "success_rate": 0.50,
            "path_planning": 0.55,
        },
    ),
    
    # ========================================================================
    # 维度 5：语言指令泛化
    # ========================================================================
    "5.1": ExperimentConfig(
        id="5.1",
        name="基线：原始指令",
        dimension="language",
        difficulty="easy",
        priority="P0",
        description="原始语言指令作为基线",
        task_description="Grab the cube and place it into the bin.",
        num_episodes=20,
        dataset_suffix="_original_instruction",
        modifications={},
        expected_metrics={
            "success_rate": 0.90,
        },
    ),
    
    "5.2": ExperimentConfig(
        id="5.2",
        name="改写指令",
        dimension="language",
        difficulty="easy",
        priority="P2",
        description="用不同的措辞表达相同的任务，测试语言理解鲁棒性",
        task_description="Pick up the cube and put it in the bin.",  # 改写
        num_episodes=10,
        dataset_suffix="_paraphrased",
        modifications={
            "instruction_style": "paraphrased",
        },
        expected_metrics={
            "success_rate": 0.85,
            "language_robustness": 0.90,
        },
    ),
    
    "5.3": ExperimentConfig(
        id="5.3",
        name="增加描述词",
        dimension="language",
        difficulty="medium",
        priority="P2",
        description="添加颜色等描述词，测试颜色识别能力",
        task_description="Grab the red cube and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_with_color_descriptor",
        modifications={
            "instruction_detail": "color_specified",
        },
        expected_metrics={
            "success_rate": 0.80,
            "color_recognition": 0.85,
        },
    ),
    
    "5.4": ExperimentConfig(
        id="5.4",
        name="空间描述",
        dimension="language",
        difficulty="medium",
        priority="P3",
        description="添加空间关系描述，测试空间推理能力",
        task_description="Grab the cube on the left and place it into the bin.",
        num_episodes=10,
        dataset_suffix="_spatial_descriptor",
        modifications={
            "instruction_detail": "spatial_specified",
        },
        expected_metrics={
            "success_rate": 0.70,
            "spatial_reasoning": 0.75,
        },
    ),
    
    "5.5": ExperimentConfig(
        id="5.5",
        name="复杂指令",
        dimension="language",
        difficulty="hard",
        priority="P3",
        description="多步骤复杂指令，测试复杂推理能力",
        task_description="Avoid the blue cube, grab the red one, and place it gently into the bin.",
        num_episodes=10,
        dataset_suffix="_complex_instruction",
        modifications={
            "instruction_complexity": "multi_step",
        },
        expected_metrics={
            "success_rate": 0.50,
            "complex_reasoning": 0.55,
        },
    ),
}


# ============================================================================
# 实验阶段定义
# ============================================================================

PHASES = {
    "1": {
        "name": "阶段 1：基础泛化（必做）",
        "experiments": ["1.1", "2.1", "3.1", "1.2", "2.3"],
    },
    "2": {
        "name": "阶段 2：进阶泛化（推荐）",
        "experiments": ["3.2", "3.3", "1.3", "2.5", "5.2"],
    },
    "3": {
        "name": "阶段 3：高级泛化（可选）",
        "experiments": ["4.2", "3.4", "4.3", "5.4", "1.4"],
    },
}


# ============================================================================
# 实验运行逻辑
# ============================================================================

def run_experiment(
    exp_config: ExperimentConfig,
    policy_path: str,
    base_dataset_repo: str = "Sprinng/eval_transfer_cube_to_bin",
    dry_run: bool = False,
):
    """运行单个泛化实验"""
    
    # 生成数据集 repo ID
    dataset_repo_id = f"{base_dataset_repo}{exp_config.dataset_suffix}"
    
    # 构建命令
    cmd = [
        "lerobot-record",
        f"--robot.type=piper_follower",
        f"--robot.id=02",
        f"--robot.control_mode=policy",
        f"--policy.path={policy_path}",
        f"--dataset.repo_id={dataset_repo_id}",
        f'--dataset.single_task="{exp_config.task_description}"',
        f"--dataset.num_episodes={exp_config.num_episodes}",
        f"--dataset.episode_time_s=60",
        f"--dataset.reset_time_s=30",
        f"--dataset.fps=30",
        f"--dataset.video=true",
        f"--dataset.push_to_hub=false",
        "--dataset.rename_map='{\"observation.images.top\":\"observation.images.camera1\", \"observation.images.wrist\":\"observation.images.camera2\", \"observation.images.side\":\"observation.images.camera3\"}'",
        f"--display_data=true",
        f"--play_sounds=true",
    ]
    
    # 打印实验信息
    print("\n" + "=" * 80)
    print(f"🧪 实验 {exp_config.id}: {exp_config.name}")
    print("=" * 80)
    print(f"\n📋 实验信息:")
    print(f"  维度:             {exp_config.dimension}")
    print(f"  难度:             {exp_config.difficulty} ({'⭐' * {'easy': 1, 'medium': 2, 'hard': 3}[exp_config.difficulty]})")
    print(f"  优先级:           {exp_config.priority}")
    print(f"  描述:             {exp_config.description}")
    print(f"\n🎯 任务指令:")
    print(f"  {exp_config.task_description}")
    print(f"\n🔧 实验修改:")
    for key, value in exp_config.modifications.items():
        print(f"  {key}: {value}")
    print(f"\n📊 预期指标:")
    for metric, value in exp_config.expected_metrics.items():
        print(f"  {metric}: {value:.2%}")
    print(f"\n💾 数据集:")
    print(f"  {dataset_repo_id}")
    print(f"\n🚀 命令:")
    print("  " + " \\\n    ".join(cmd))
    print("\n" + "=" * 80)
    
    if dry_run:
        print("\n✅ Dry run - 命令已打印\n")
        return
    
    # 确认运行
    print(f"\n⏳ 即将开始实验 {exp_config.id}，3秒后启动... (Ctrl+C 取消)")
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ 实验已取消\n")
        return
    
    # 运行实验
    print(f"\n🏃 运行实验 {exp_config.id}...\n")
    try:
        cmd_str = " ".join(cmd)
        result = subprocess.run(cmd_str, shell=True, cwd=project_root)
        if result.returncode == 0:
            print(f"\n✅ 实验 {exp_config.id} 完成！\n")
        else:
            print(f"\n❌ 实验 {exp_config.id} 失败（返回码: {result.returncode}）\n")
    except KeyboardInterrupt:
        print(f"\n\n⚠️  实验 {exp_config.id} 被用户中断\n")


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA 泛化性能评估实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有实验
  python myscripts/eval/eval_smolvla_generalization.py --list
  
  # 运行单个实验
  python myscripts/eval/eval_smolvla_generalization.py --experiment 1.1
  
  # 运行整个阶段的实验
  python myscripts/eval/eval_smolvla_generalization.py --phase 1
  
  # 运行所有 P0 优先级的实验
  python myscripts/eval/eval_smolvla_generalization.py --priority P0
  
  # Dry run（只打印命令）
  python myscripts/eval/eval_smolvla_generalization.py --experiment 1.1 --dry-run
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的实验"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="运行指定的实验（例如：1.1, 2.3）"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3"],
        help="运行整个阶段的实验"
    )
    parser.add_argument(
        "--priority",
        type=str,
        choices=["P0", "P1", "P2", "P3"],
        help="运行指定优先级的所有实验"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有实验"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="outputs/train/piper_smolvla_transfer_cube_to_bin/checkpoints/last/pretrained_model",
        help="策略模型路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run（只打印命令，不执行）"
    )
    
    args = parser.parse_args()
    
    # 列出实验
    if args.list:
        print("\n" + "=" * 80)
        print("📋 可用的泛化评估实验")
        print("=" * 80)
        
        for dimension in ["object", "spatial", "environment", "manipulation", "language"]:
            dimension_name = {
                "object": "维度 1：物体属性泛化",
                "spatial": "维度 2：目标位置泛化",
                "environment": "维度 3：环境干扰泛化",
                "manipulation": "维度 4：操作模式泛化",
                "language": "维度 5：语言指令泛化",
            }[dimension]
            
            print(f"\n{dimension_name}:")
            for exp_id, exp in EXPERIMENTS.items():
                if exp.dimension == dimension:
                    difficulty_stars = "⭐" * {"easy": 1, "medium": 2, "hard": 3}[exp.difficulty]
                    print(f"  [{exp.priority}] {exp.id} - {exp.name} {difficulty_stars}")
                    print(f"      {exp.description}")
        
        print("\n" + "=" * 80)
        print("\n阶段定义:")
        for phase_id, phase_info in PHASES.items():
            print(f"\n阶段 {phase_id}: {phase_info['name']}")
            print(f"  实验: {', '.join(phase_info['experiments'])}")
        
        print("\n" + "=" * 80)
        return
    
    # 确定要运行的实验
    experiments_to_run = []
    
    if args.experiment:
        if args.experiment in EXPERIMENTS:
            experiments_to_run = [args.experiment]
        else:
            print(f"❌ 实验 {args.experiment} 不存在")
            sys.exit(1)
    
    elif args.phase:
        experiments_to_run = PHASES[args.phase]["experiments"]
        print(f"\n🎯 运行 {PHASES[args.phase]['name']}")
    
    elif args.priority:
        experiments_to_run = [
            exp_id for exp_id, exp in EXPERIMENTS.items()
            if exp.priority == args.priority
        ]
        print(f"\n🎯 运行所有 {args.priority} 优先级的实验")
    
    elif args.all:
        experiments_to_run = list(EXPERIMENTS.keys())
        print(f"\n🎯 运行所有实验")
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # 运行实验
    print(f"\n共 {len(experiments_to_run)} 个实验\n")
    
    for i, exp_id in enumerate(experiments_to_run, 1):
        print(f"\n{'=' * 80}")
        print(f"进度: {i}/{len(experiments_to_run)}")
        print(f"{'=' * 80}")
        
        exp_config = EXPERIMENTS[exp_id]
        run_experiment(
            exp_config=exp_config,
            policy_path=args.policy_path,
            dry_run=args.dry_run,
        )
    
    print(f"\n🎉 所有实验完成！")


if __name__ == "__main__":
    main()