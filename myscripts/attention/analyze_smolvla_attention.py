"""
Detailed SmolVLA Attention Analysis

This script provides in-depth analysis of attention patterns in SmolVLA,
including per-head visualization, attention statistics, and pattern analysis.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

sys.path.insert(0, "/home/zwt/Projects/lerobot")
sys.path.insert(0, "/home/zwt/Projects/lerobot/src")

from transformers import AutoProcessor

# 正确的导入路径
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


@dataclass
class DetailedAttentionConfig:
    """Extended configuration for detailed attention analysis"""
    model_path: str = "lerobot/smolvla_base"
    input_dir: str = "./myscripts/attention/inputs"
    output_dir: str = "./myscripts/attention/outputs"
    analyze_heads: bool = True
    compute_statistics: bool = True
    save_attention_data: bool = True
    visualization_dpi: int = 150
    selected_layers: Optional[List[int]] = None  # None means all layers


class SmolVLAAttentionHook:
    """Hook class to capture attention weights from SmolVLA model"""
    
    def __init__(self):
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.hooks = []
    
    def clear(self):
        self.attention_weights = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _make_hook(self, name: str):
        """Create a hook function that captures attention weights"""
        def hook_fn(module, input, output):
            # SmolVLA uses eager_attention_forward which computes attention internally
            # We need to hook into the attention computation
            pass
        return hook_fn
    
    def _make_attention_capture_hook(self, name: str):
        """
        Create a hook to capture the attention weights computed in eager_attention_forward.
        We'll modify the forward to store attention weights.
        """
        def hook_fn(module, args, kwargs, output):
            # This captures output after attention layer
            if isinstance(output, torch.Tensor):
                self.attention_weights[name] = output.detach().cpu()
        return hook_fn


class SmolVLADetailedAnalyzer:
    """Detailed analyzer for SmolVLA attention patterns"""
    
    def __init__(self, config: DetailedAttentionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None
        self.processor = None
        self.attention_cache: Dict[str, Any] = {}
        self.current_output_dir: str = config.output_dir
        
        # Create input and output directories
        os.makedirs(config.input_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _get_output_dir_for_image(self, image_path: Optional[str] = None, image_name: Optional[str] = None) -> str:
        """
        Get output directory based on input image name.
        
        Args:
            image_path: Full path to the image file
            image_name: Name to use for the output folder (without extension)
        
        Returns:
            Path to the output directory for this image
        """
        if image_name:
            folder_name = image_name
        elif image_path:
            # Extract filename without extension
            folder_name = Path(image_path).stem
        else:
            folder_name = "default"
        
        output_dir = os.path.join(self.config.output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        self.current_output_dir = output_dir
        return output_dir
    
    def load_model(self):
        """Load SmolVLA policy model"""
        print(f"Loading SmolVLA from {self.config.model_path}...")
        
        # Load the policy using the pretrained method
        self.policy = SmolVLAPolicy.from_pretrained(self.config.model_path)
        self.policy.to(self.device)
        self.policy.eval()
        
        # Get processor from the VLM
        self.processor = self.policy.model.vlm_with_expert.processor
        
        # Print model architecture info
        self._print_model_info()
    
    def _print_model_info(self):
        """Print relevant model architecture information"""
        print("\n=== SmolVLA Model Architecture ===")
        
        vlm_with_expert = self.policy.model.vlm_with_expert
        
        # Vision encoder info
        vision_model = vlm_with_expert.get_vlm_model().vision_model
        print(f"Vision Encoder: {type(vision_model).__name__}")
        if hasattr(vision_model, 'config'):
            vc = vision_model.config
            print(f"  - Hidden size: {getattr(vc, 'hidden_size', 'N/A')}")
            print(f"  - Num layers: {getattr(vc, 'num_hidden_layers', 'N/A')}")
            print(f"  - Num heads: {getattr(vc, 'num_attention_heads', 'N/A')}")
        
        # VLM text model info
        text_model = vlm_with_expert.get_vlm_model().text_model
        print(f"\nVLM Text Model: {type(text_model).__name__}")
        print(f"  - Num VLM layers: {vlm_with_expert.num_vlm_layers}")
        print(f"  - Num attention heads: {vlm_with_expert.num_attention_heads}")
        print(f"  - Num KV heads: {vlm_with_expert.num_key_value_heads}")
        
        # Expert model info
        expert = vlm_with_expert.lm_expert
        print(f"\nAction Expert: {type(expert).__name__}")
        print(f"  - Num expert layers: {vlm_with_expert.num_expert_layers}")
        print(f"  - Expert hidden size: {vlm_with_expert.expert_hidden_size}")
        print(f"  - Attention mode: {vlm_with_expert.attention_mode}")
        
        print("=" * 40 + "\n")
    
    def load_image_from_input(self, image_name: str) -> Image.Image:
        """
        Load an image from the input directory.
        
        Args:
            image_name: Name of the image file (e.g., "cube.png")
        
        Returns:
            PIL Image object
        """
        image_path = os.path.join(self.config.input_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}")
        
        # Set up output directory based on image name
        self._get_output_dir_for_image(image_path=image_path)
        
        return image
    
    def extract_attention_with_hooks(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[torch.Tensor] = None,
        image_path: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights by patching the attention forward function.
        
        Args:
            image: PIL Image
            instruction: Task instruction string
            state: Optional state tensor
            image_path: Optional path to image (used for output folder naming)
            image_name: Optional name for output folder (without extension)
        """
        # Set up output directory
        if image_path or image_name:
            self._get_output_dir_for_image(image_path=image_path, image_name=image_name)
        
        attention_dict = {}
        original_eager_attention = None
        
        vlm_with_expert = self.policy.model.vlm_with_expert
        
        # Patch the eager_attention_forward to capture attention weights
        def patched_attention_forward(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        ):
            num_att_heads = vlm_with_expert.num_attention_heads
            num_key_value_heads = vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            
            sequence_length = key_states.shape[1]
            
            key_states_expanded = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states_expanded = key_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )
            
            value_states_expanded = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            value_states_expanded = value_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )
            
            query_states_f32 = query_states.to(dtype=torch.float32)
            key_states_f32 = key_states_expanded.to(dtype=torch.float32)
            
            query_states_t = query_states_f32.transpose(1, 2)
            key_states_t = key_states_f32.transpose(1, 2)
            
            att_weights = torch.matmul(query_states_t, key_states_t.transpose(2, 3))
            att_weights *= head_dim**-0.5
            
            att_weights = att_weights.to(dtype=torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
            
            # Store the attention probabilities
            layer_idx = len(attention_dict)
            attention_dict[f"layer_{layer_idx}"] = probs.detach().cpu()
            
            probs = probs.to(dtype=value_states_expanded.dtype)
            att_output = torch.matmul(probs, value_states_expanded.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )
            
            return att_output
        
        # Prepare inputs
        images, img_masks, lang_tokens, lang_masks, state_tensor = self._prepare_inputs(
            image, instruction, state
        )
        
        # Patch and run forward
        original_eager_attention = vlm_with_expert.eager_attention_forward
        vlm_with_expert.eager_attention_forward = patched_attention_forward
        
        try:
            with torch.no_grad():
                # Sample actions (this triggers the forward pass)
                _ = self.policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state_tensor
                )
        finally:
            # Restore original function
            vlm_with_expert.eager_attention_forward = original_eager_attention
        
        self.attention_cache = attention_dict
        return attention_dict
    
    def _prepare_inputs(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Prepare inputs for SmolVLA model"""
        # Resize image if needed
        target_size = self.policy.config.resize_imgs_with_padding
        if target_size:
            image = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        
        # Convert image to tensor
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Normalize from [0,1] to [-1,1]
        img_tensor = img_tensor * 2.0 - 1.0
        
        images = [img_tensor]
        img_masks = [torch.ones(1, dtype=torch.bool, device=self.device)]
        
        # Tokenize instruction
        if not instruction.endswith("\n"):
            instruction = instruction + "\n"
        
        tokens = self.processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True
        )
        lang_tokens = tokens["input_ids"].to(self.device)
        lang_masks = tokens["attention_mask"].bool().to(self.device)
        
        # Prepare state
        if state is None:
            state = torch.zeros(1, self.policy.config.max_state_dim, device=self.device)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            # Pad state if needed
            if state.shape[-1] < self.policy.config.max_state_dim:
                padding = torch.zeros(
                    state.shape[0], 
                    self.policy.config.max_state_dim - state.shape[-1],
                    device=self.device
                )
                state = torch.cat([state, padding], dim=-1)
        
        return images, img_masks, lang_tokens, lang_masks, state
    
    def compute_attention_statistics(
        self,
        attention_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for attention patterns"""
        stats = {}
        
        for name, attn in attention_weights.items():
            if attn is None:
                continue
            
            attn_np = attn.float().numpy()
            flat_attn = attn_np.flatten()
            
            # Avoid log(0)
            flat_attn_safe = np.clip(flat_attn, 1e-10, 1.0)
            
            stats[name] = {
                'mean': float(np.mean(flat_attn)),
                'std': float(np.std(flat_attn)),
                'max': float(np.max(flat_attn)),
                'min': float(np.min(flat_attn)),
                'entropy': float(-np.sum(flat_attn_safe * np.log(flat_attn_safe))),
                'sparsity': float(np.mean(flat_attn < 0.01)),
                'shape': list(attn.shape),
            }
            
            # Per-head statistics
            if len(attn.shape) >= 3:
                n_heads = attn.shape[1] if len(attn.shape) == 4 else attn.shape[0]
                head_means = []
                for h in range(min(n_heads, 32)):  # Limit to first 32 heads
                    if len(attn.shape) == 4:
                        head_attn = attn_np[0, h]
                    else:
                        head_attn = attn_np[h]
                    head_means.append(float(np.mean(head_attn)))
                stats[name]['head_means'] = head_means
        
        return stats
    
    def visualize_attention_heatmap(
        self,
        attention: torch.Tensor,
        title: str,
        save_path: Optional[str] = None,
        head_idx: Optional[int] = None
    ):
        """Visualize a single attention heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Process attention tensor
        if len(attention.shape) == 4:
            # [batch, heads, seq, seq]
            if head_idx is not None:
                attn_2d = attention[0, head_idx].float().numpy()
            else:
                attn_2d = attention[0].mean(0).float().numpy()
        elif len(attention.shape) == 3:
            if head_idx is not None:
                attn_2d = attention[head_idx].float().numpy()
            else:
                attn_2d = attention.mean(0).float().numpy()
        else:
            attn_2d = attention.float().numpy()
        
        im = ax.imshow(attn_2d, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Key Position", fontsize=12)
        ax.set_ylabel("Query Position", fontsize=12)
        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def visualize_per_head_attention(
        self,
        attention_weights: Dict[str, torch.Tensor],
        layer_name: str,
        save_path: Optional[str] = None
    ):
        """Visualize attention for each head in a layer"""
        if layer_name not in attention_weights:
            print(f"Layer {layer_name} not found in attention weights")
            return
        
        attn = attention_weights[layer_name]
        
        if len(attn.shape) == 4:
            attn = attn[0]  # Remove batch: [heads, seq, seq]
        
        n_heads = min(attn.shape[0], 16)  # Limit display
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f"Per-Head Attention: {layer_name}", fontsize=14)
        
        for h in range(n_heads):
            row = h // n_cols
            col = h % n_cols
            ax = axes[row, col]
            
            head_attn = attn[h].float().numpy()
            
            im = ax.imshow(head_attn, cmap='viridis', aspect='auto')
            ax.set_title(f"Head {h}", fontsize=10)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for h in range(n_heads, n_rows * n_cols):
            row = h // n_cols
            col = h % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()
    
    def visualize_attention_flow(
        self,
        attention_weights: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ):
        """Visualize how attention patterns change across layers"""
        layers = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[-1]))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean attention per layer
        ax = axes[0]
        layer_means = []
        layer_names = []
        
        for name in layers:
            attn = attention_weights[name]
            if len(attn.shape) == 4:
                mean_attn = attn[0].mean().item()
            else:
                mean_attn = attn.mean().item()
            layer_means.append(mean_attn)
            layer_names.append(name.replace('layer_', 'L'))
        
        ax.bar(layer_names, layer_means, color='steelblue')
        ax.set_title("Mean Attention per Layer", fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Attention")
        ax.tick_params(axis='x', rotation=45)
        
        # Attention entropy per layer
        ax = axes[1]
        layer_entropies = []
        
        for name in layers:
            attn = attention_weights[name]
            attn_np = attn.float().numpy().flatten()
            attn_safe = np.clip(attn_np, 1e-10, 1.0)
            entropy = -np.sum(attn_safe * np.log(attn_safe))
            layer_entropies.append(entropy)
        
        ax.bar(layer_names, layer_entropies, color='coral')
        ax.set_title("Attention Entropy per Layer", fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy")
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()
    
    def _create_layer_grid_visualization(
        self,
        attention_weights: Dict[str, torch.Tensor],
        save_path: str
    ):
        """Create a grid visualization of all layers similar to reference images"""
        all_layers = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[-1]))
        n_layers = len(all_layers)
        
        if n_layers == 0:
            print("No attention layers to visualize")
            return
        
        n_cols = min(6, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle("SmolVLA Attention Heatmaps Across Layers", fontsize=14, y=1.02)
        
        for idx, layer_name in enumerate(all_layers):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            attn = attention_weights[layer_name]
            
            # Process to 2D (average over batch and heads)
            if len(attn.shape) == 4:
                attn_2d = attn[0].mean(0).float().numpy()
            elif len(attn.shape) == 3:
                attn_2d = attn.mean(0).float().numpy()
            else:
                attn_2d = attn.float().numpy()
            
            im = ax.imshow(attn_2d, cmap='magma', aspect='auto')
            
            title = layer_name.replace('_', ' ').title()
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=5)
        
        # Hide empty subplots
        for idx in range(n_layers, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap='magma')
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Attention')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.98])
        fig.savefig(save_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    
    def create_comprehensive_report(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[np.ndarray] = None,
        prefix: str = "smolvla_analysis",
        image_path: Optional[str] = None,
        image_name: Optional[str] = None
    ):
        """Create a comprehensive attention analysis report"""
        # Set up output directory
        if image_path or image_name:
            self._get_output_dir_for_image(image_path=image_path, image_name=image_name)
        
        print(f"Saving outputs to: {self.current_output_dir}")
        print("Extracting attention weights...")
        
        attention_weights = self.extract_attention_with_hooks(
            image, instruction, state,
            image_path=image_path, image_name=image_name
        )
        
        if not attention_weights:
            print("Warning: No attention weights extracted. Check model compatibility.")
            return
        
        print(f"Extracted attention from {len(attention_weights)} layers")
        for name, attn in attention_weights.items():
            print(f"  {name}: {attn.shape}")
        
        # Compute statistics
        if self.config.compute_statistics:
            print("Computing attention statistics...")
            stats = self.compute_attention_statistics(attention_weights)
            
            stats_path = os.path.join(self.current_output_dir, f"{prefix}_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics: {stats_path}")
        
        # Create visualizations
        print("Generating visualizations...")
        
        # 1. Layer grid
        self._create_layer_grid_visualization(
            attention_weights,
            os.path.join(self.current_output_dir, f"{prefix}_layer_grid.png")
        )
        
        # 2. Attention flow
        self.visualize_attention_flow(
            attention_weights,
            os.path.join(self.current_output_dir, f"{prefix}_attention_flow.png")
        )
        
        # 3. Per-head analysis for selected layers
        if self.config.analyze_heads:
            layers_to_analyze = list(attention_weights.keys())[:3]
            for layer_name in layers_to_analyze:
                self.visualize_per_head_attention(
                    attention_weights,
                    layer_name,
                    os.path.join(self.current_output_dir, f"{prefix}_{layer_name}_heads.png")
                )
        
        # 4. Save raw attention data
        if self.config.save_attention_data:
            data_path = os.path.join(self.current_output_dir, f"{prefix}_attention_data.pt")
            torch.save(attention_weights, data_path)
            print(f"Saved attention data: {data_path}")
        
        print(f"\nAnalysis complete! Check {self.current_output_dir}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.policy is not None:
            del self.policy
        torch.cuda.empty_cache()


def main():
    """Run detailed attention analysis"""
    config = DetailedAttentionConfig(
        model_path="lerobot/smolvla_base",
        input_dir="./myscripts/attention/inputs",
        output_dir="./myscripts/attention/outputs",
        analyze_heads=True,
        compute_statistics=True,
        save_attention_data=True,
    )
    
    analyzer = SmolVLADetailedAnalyzer(config)
    
    try:
        analyzer.load_model()
        
        # Check if there are images in input directory
        input_images = list(Path(config.input_dir).glob("*.png")) + \
                       list(Path(config.input_dir).glob("*.jpg")) + \
                       list(Path(config.input_dir).glob("*.jpeg"))
        
        if input_images:
            # Process first image found
            image_path = str(input_images[0])
            image = Image.open(image_path).convert('RGB')
            image_name = Path(image_path).stem
            print(f"Processing image: {image_path}")
        else:
            # Use default test image
            image = Image.new('RGB', (640, 480), color=(100, 150, 200))
            image_name = "test_image"
            print("No images in input folder, using test image")
        
        instruction = "Pick up the red cube and place it on the blue plate."
        
        analyzer.create_comprehensive_report(
            image,
            instruction,
            prefix="smolvla_detailed",
            image_name=image_name
        )
        
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()