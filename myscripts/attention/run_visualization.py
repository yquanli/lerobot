"""
SmolVLA Attention Visualization - 命令行入口

使用方法:
    # 使用数据集
    python run_visualization.py --repo-id Sprinng/piper_transfer_cube_to_bin --episode-index 0 --frame-index 50
    
    # 使用本地图片
    python run_visualization.py --image cube.png --instruction "Pick up the cube"
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, "/home/zwt/Projects/lerobot")
sys.path.insert(0, "/home/zwt/Projects/lerobot/src")

import torch
import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def get_input_dir():
    return "./myscripts/attention/inputs"


def get_output_dir():
    return "./myscripts/attention/outputs"


def find_images_in_input():
    """Find all images in the input directory"""
    input_dir = Path(get_input_dir())
    input_dir.mkdir(parents=True, exist_ok=True)
    
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        images.extend(input_dir.glob(ext))
    return sorted(images)


def load_dataset_item(
    repo_id: str,
    episode_index: int = 0,
    frame_index: int = 0,
    root: Optional[str] = None
) -> Dict:
    """从 LeRobotDataset 加载数据项"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode_index] if episode_index is not None else None,
    )
    
    ep_meta = dataset.meta.episodes[episode_index]
    ep_start = ep_meta['dataset_from_index']
    ep_end = ep_meta['dataset_to_index']
    ep_length = ep_end - ep_start
    
    if frame_index >= ep_length:
        print(f"Warning: frame_index {frame_index} exceeds episode length {ep_length}, using last frame")
        frame_index = ep_length - 1
    
    global_idx = ep_start + frame_index
    print(f"Episode {episode_index}: frames {ep_start}-{ep_end} (length={ep_length})")
    print(f"Loading frame {frame_index} (global index: {global_idx})")
    
    item = dataset[global_idx]
    
    images = {}
    for key in dataset.meta.camera_keys:
        if key in item:
            img_tensor = item[key]
            if isinstance(img_tensor, torch.Tensor):
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                images[key] = Image.fromarray(img_np)
    
    state = None
    if 'observation.state' in item:
        state = item['observation.state']
        if isinstance(state, torch.Tensor):
            if state.ndim == 2:
                state = state[0]
    
    instruction = None
    if 'task' in item:
        instruction = item['task']
        if isinstance(instruction, list):
            instruction = instruction[0]
    
    if instruction is None:
        tasks = dataset.meta.tasks
        if hasattr(tasks, 'index') and len(tasks) > 0:
            instruction = tasks.index[0]
        else:
            instruction = "Perform the task."
    
    return {
        'images': images,
        'state': state,
        'instruction': instruction,
        'stats': dataset.meta.stats,
        'episode_index': episode_index,
        'frame_index': frame_index,
        'camera_keys': list(images.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Attention Visualization")
    
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repository ID")
    parser.add_argument("--dataset-root", type=str, default=None, help="Local dataset root")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--instruction", type=str, default="Pick up the red object.", help="Task instruction")
    parser.add_argument("--model", type=str, default="lerobot/smolvla_base", help="Model path")
    parser.add_argument("--output", type=str, default="./myscripts/attention/outputs", help="Output directory")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Overlay transparency")
    parser.add_argument("--save-matrix", action="store_true", help="Save raw attention matrices")
    
    args = parser.parse_args()
    
    input_dir = get_input_dir()
    output_dir = args.output if args.output else get_output_dir()
    
    # Ensure directories exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    from visualize_attention import SmolVLAAttentionVisualizer, AttentionConfig
    
    config = AttentionConfig(
        model_path=args.model,
        input_dir=input_dir,
        output_dir=output_dir,
        overlay_alpha=args.overlay_alpha,
        save_raw_attention_matrix=args.save_matrix,
    )
    
    visualizer = SmolVLAAttentionVisualizer(config)
    
    try:
        visualizer.load_model()
        
        if args.repo_id:
            print("\n" + "=" * 60)
            print("Loading data from LeRobotDataset")
            print("=" * 60)
            
            data = load_dataset_item(
                repo_id=args.repo_id,
                episode_index=args.episode_index,
                frame_index=args.frame_index,
                root=args.dataset_root
            )
            
            images = data['images']
            state = data['state']
            instruction = data['instruction']
            
            visualizer.set_normalization_stats(data['stats'])
            image_name = f"ep{args.episode_index:03d}_frame{args.frame_index:04d}"
            
            print(f"\nData loaded:")
            print(f"  Cameras: {list(images.keys())}")
            print(f"  State shape: {state.shape if state is not None else 'None'}")
            print(f"  Instruction: {instruction}")
            
            print(f"\n{'=' * 50}")
            print(f"Processing: {image_name}")
            print(f"{'=' * 50}")
            
            attention_weights, actions = visualizer.extract_attention_with_output(
                images=images,
                instruction=instruction,
                state=state,
                image_name=image_name,
                normalize_state=True
            )
            
            print(f"Captured attention from {len(attention_weights)} layers")
            print(f"Predicted actions shape: {actions.shape}")
            
            visualizer.visualize_all(attention_weights, images, instruction)
            
        elif args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                image_path = Path("./myscripts/attention/inputs") / args.image
            
            if not image_path.exists():
                print(f"Image not found: {args.image}")
                return
            
            image = Image.open(image_path).convert('RGB')
            images = {"camera": image}
            state = None
            instruction = args.instruction
            image_name = image_path.stem
            
            print(f"\n{'=' * 50}")
            print(f"Processing: {image_name}")
            print(f"{'=' * 50}")
            
            attention_weights, _ = visualizer.extract_attention_with_output(
                images=images,
                instruction=instruction,
                image_name=image_name
            )
            
            visualizer.visualize_all_layers(attention_weights, images, instruction)
        
        else:
            images_to_process = find_images_in_input()
            if not images_to_process:
                print(f"No images in {input_dir}, using random test image")
                img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                test_image = Image.fromarray(img_array)
                images_to_process = [(test_image, "test_image")]
            
            for item in images_to_process:
                if isinstance(item, tuple):
                    image, image_name = item
                else:
                    image_path = str(item)
                    image = Image.open(image_path).convert('RGB')
                    image_name = item.stem
                
                print(f"\n{'=' * 50}")
                print(f"Processing: {image_name}")
                print(f"{'=' * 50}")
                
                images = {"camera": image}
                
                attention_weights, _ = visualizer.extract_attention_with_output(
                    images=images,
                    instruction=args.instruction,
                    image_name=image_name
                )
                
                visualizer.visualize_all_layers(attention_weights, images, args.instruction)
        
        print(f"\nDone! Check {output_dir} for results.")
        print("\nGenerated files include:")
        print("  - overlays/vlm/         : VLM layer attention overlays (Layer 0-15)")
        print("  - overlays/denoising/   : Denoising cross-attention overlays")
        print("  - comparison/           : Multi-camera comparison (if multiple cameras)")
        print("  - *_overview.png        : Grid overview images")
        if args.save_matrix:
            print("  - raw_matrices/         : Raw attention matrices")
        
    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    main()