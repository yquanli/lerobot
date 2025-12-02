"""
Quick script to run SmolVLA attention visualization.

Usage:
    python run_visualization.py --image path/to/image.jpg --instruction "your instruction"
    python run_visualization.py --image cube.png  # Load from input folder
    python run_visualization.py  # Process all images in input folder
    python run_visualization.py --detailed  # Run detailed analysis
    python run_visualization.py --overlay-only  # Only generate overlay visualizations
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/zwt/Projects/lerobot")
sys.path.insert(0, "/home/zwt/Projects/lerobot/src")

from PIL import Image
import numpy as np


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


def main():
    parser = argparse.ArgumentParser(description="Visualize SmolVLA Attention")
    parser.add_argument("--image", type=str, default=None, 
                        help="Path to input image or image name in input folder")
    parser.add_argument("--instruction", type=str, default="Pick up the red object.", 
                        help="Task instruction")
    parser.add_argument("--model", type=str, default="lerobot/smolvla_base", 
                        help="Model path")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output directory (default: ./myscripts/attention/outputs)")
    parser.add_argument("--detailed", action="store_true", 
                        help="Run detailed analysis")
    parser.add_argument("--all", action="store_true",
                        help="Process all images in input folder")
    parser.add_argument("--overlay-alpha", type=float, default=0.5,
                        help="Attention overlay transparency (0-1)")
    parser.add_argument("--overlay-cmap", type=str, default="jet",
                        choices=["jet", "hot", "viridis", "plasma", "inferno", "magma"],
                        help="Colormap for attention overlay")
    parser.add_argument("--layers", type=str, default=None,
                        help="Specific layers to visualize, e.g., '15,16,17,18'")
    
    args = parser.parse_args()
    
    input_dir = get_input_dir()
    output_dir = args.output if args.output else get_output_dir()
    
    # Ensure directories exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse specified layers
    selected_layers = None
    if args.layers:
        try:
            selected_layers = [int(x.strip()) for x in args.layers.split(',')]
        except ValueError:
            print(f"Invalid layers format: {args.layers}")
            print("Expected format: '15,16,17,18'")
            return
    
    # Collect images to process
    images_to_process = []
    
    if args.all:
        images_to_process = find_images_in_input()
        if not images_to_process:
            print(f"No images found in {input_dir}")
            print("Please add images to the input folder or specify --image")
            return
    elif args.image:
        image_path = Path(args.image)
        
        if image_path.exists():
            images_to_process = [image_path]
        else:
            input_path = Path(input_dir) / args.image
            if input_path.exists():
                images_to_process = [input_path]
            else:
                print(f"Image not found: {args.image}")
                print(f"Also checked: {input_path}")
                return
    else:
        images_to_process = find_images_in_input()
        if not images_to_process:
            print(f"No images in {input_dir}, using random test image")
            img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_image = Image.fromarray(img_array)
            images_to_process = [(test_image, "test_image")]
    
    # Process images
    if args.detailed:
        from analyze_smolvla_attention import SmolVLADetailedAnalyzer, DetailedAttentionConfig
        
        config = DetailedAttentionConfig(
            model_path=args.model,
            input_dir=input_dir,
            output_dir=output_dir,
            analyze_heads=True,
            compute_statistics=True,
            save_attention_data=True,
        )
        
        analyzer = SmolVLADetailedAnalyzer(config)
        try:
            analyzer.load_model()
            
            for item in images_to_process:
                if isinstance(item, tuple):
                    image, image_name = item
                    image_path = None
                else:
                    image_path = str(item)
                    image = Image.open(image_path).convert('RGB')
                    image_name = item.stem
                
                print(f"\n{'='*50}")
                print(f"Processing: {image_name}")
                print(f"{'='*50}")
                
                analyzer.create_comprehensive_report(
                    image, 
                    args.instruction,
                    image_name=image_name
                )
        finally:
            analyzer.cleanup()
    else:
        from visualize_attention import SmolVLAAttentionVisualizer, AttentionConfig
        
        config = AttentionConfig(
            model_path=args.model,
            input_dir=input_dir,
            output_dir=output_dir,
            overlay_alpha=args.overlay_alpha,
            overlay_cmap=args.overlay_cmap,
            selected_layers_for_overlay=selected_layers,
        )
        
        visualizer = SmolVLAAttentionVisualizer(config)
        try:
            visualizer.load_model()
            
            for item in images_to_process:
                if isinstance(item, tuple):
                    image, image_name = item
                else:
                    image_path = str(item)
                    image = Image.open(image_path).convert('RGB')
                    image_name = item.stem
                
                print(f"\n{'='*50}")
                print(f"Processing: {image_name}")
                print(f"{'='*50}")
                
                attention_weights, _ = visualizer.extract_attention_with_output(
                    image, args.instruction, image_name=image_name
                )
                
                # Full visualization
                visualizer.visualize_all_layers(attention_weights, image, args.instruction)
                
                # If specific layers are selected, generate overlay for those layers
                if selected_layers:
                    visualizer.visualize_attention_on_image(
                        attention_weights, image, args.instruction,
                        selected_layers=selected_layers,
                        prefix="selected_layers_overlay"
                    )
        finally:
            visualizer.cleanup()
    
    print(f"\nDone! Check {output_dir} for results.")
    print("\nGenerated files include:")
    print("  - *_layer_grid.png: Attention heatmaps for all layers")
    print("  - *_overview.png: Overview with image and selected layers")
    print("  - *_overlay_grid.png: Attention overlaid on original image")
    print("  - *_comparison_paper_style.png: Paper-style visualization")
    print("  - overlays/: Individual overlay images for each layer")


if __name__ == "__main__":
    main()