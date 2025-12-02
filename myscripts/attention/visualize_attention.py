"""
SmolVLA Attention Heatmap Visualization

This script visualizes attention patterns across different layers of SmolVLA model
during inference, similar to the attention visualizations shown in other VLA models.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import cv2

sys.path.insert(0, "/home/zwt/Projects/lerobot")
sys.path.insert(0, "/home/zwt/Projects/lerobot/src")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@dataclass
class AttentionConfig:
    """Configuration for attention visualization"""
    model_path: str = "lerobot/smolvla_base"
    input_dir: str = "./myscripts/attention/inputs"
    output_dir: str = "./myscripts/attention/outputs"
    figsize: Tuple[int, int] = (20, 16)
    cmap: str = "viridis"
    save_individual: bool = True
    save_combined: bool = True
    overlay_alpha: float = 0.5  # Heatmap overlay transparency
    overlay_cmap: str = "jet"   # Colormap for overlay heatmap
    selected_layers_for_overlay: Optional[List[int]] = None  # Layers to overlay


class SmolVLAAttentionVisualizer:
    """Visualizer for SmolVLA attention patterns"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None
        self.processor = None
        self.attention_cache: Dict[str, torch.Tensor] = {}
        self.current_output_dir: str = config.output_dir
        
        # Image token related information (to be extracted from model config)
        self.image_token_info: Dict = {}
        
        # Create input and output directories
        os.makedirs(config.input_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _get_output_dir_for_image(self, image_path: Optional[str] = None, image_name: Optional[str] = None) -> str:
        """Get output directory based on input image name."""
        if image_name:
            folder_name = image_name
        elif image_path:
            folder_name = Path(image_path).stem
        else:
            folder_name = "default"
        
        output_dir = os.path.join(self.config.output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        self.current_output_dir = output_dir
        return output_dir
    
    def load_model(self):
        """Load SmolVLA model"""
        print(f"Loading model from {self.config.model_path}...")
        
        self.policy = SmolVLAPolicy.from_pretrained(self.config.model_path)
        self.policy.to(self.device)
        self.policy.eval()
        
        self.processor = self.policy.model.vlm_with_expert.processor
        
        # Extract image token configuration
        self._extract_image_token_info()
        
        print("Model loaded successfully!")
        self._print_model_info()
    
    def _extract_image_token_info(self):
        """Extract image token layout information for mapping attention back to image space"""
        vlm = self.policy.model.vlm_with_expert.get_vlm_model()
        
        # Get vision encoder configuration
        vision_config = vlm.vision_model.config if hasattr(vlm, 'vision_model') else None
        
        if vision_config:
            # Typical ViT configuration
            image_size = getattr(vision_config, 'image_size', 384)
            patch_size = getattr(vision_config, 'patch_size', 14)
            
            if isinstance(image_size, (list, tuple)):
                image_size = image_size[0]
            
            num_patches_per_side = image_size // patch_size
            num_image_tokens = num_patches_per_side * num_patches_per_side
            
            self.image_token_info = {
                'image_size': image_size,
                'patch_size': patch_size,
                'num_patches_per_side': num_patches_per_side,
                'num_image_tokens': num_image_tokens,
            }
            print(f"Image token info: {self.image_token_info}")
        else:
            # Use default values
            self.image_token_info = {
                'image_size': 384,
                'patch_size': 14,
                'num_patches_per_side': 27,  # 384/14 ≈ 27
                'num_image_tokens': 729,
            }
            print(f"Using default image token info: {self.image_token_info}")
    
    def _print_model_info(self):
        """Print model architecture info"""
        vlm = self.policy.model.vlm_with_expert
        print(f"\n=== SmolVLA Architecture ===")
        print(f"VLM layers: {vlm.num_vlm_layers}")
        print(f"Expert layers: {vlm.num_expert_layers}")
        print(f"Attention heads: {vlm.num_attention_heads}")
        print(f"Attention mode: {vlm.attention_mode}")
        print("=" * 30 + "\n")
    
    def load_image_from_input(self, image_name: str) -> Image.Image:
        """Load an image from the input directory."""
        image_path = os.path.join(self.config.input_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}")
        
        self._get_output_dir_for_image(image_path=image_path)
        return image
    
    def _prepare_inputs(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Prepare inputs for SmolVLA model"""
        target_size = self.policy.config.resize_imgs_with_padding
        if target_size:
            image = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor = img_tensor * 2.0 - 1.0
        
        images = [img_tensor]
        img_masks = [torch.ones(1, dtype=torch.bool, device=self.device)]
        
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
        
        if state is None:
            state = torch.zeros(1, self.policy.config.max_state_dim, device=self.device)
        
        return images, img_masks, lang_tokens, lang_masks, state
    
    def extract_attention_with_output(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[torch.Tensor] = None,
        image_path: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run forward pass and extract attention weights."""
        if image_path or image_name:
            self._get_output_dir_for_image(image_path=image_path, image_name=image_name)
        
        attention_dict = {}
        vlm_with_expert = self.policy.model.vlm_with_expert
        
        original_eager_attention = vlm_with_expert.eager_attention_forward
        
        def patched_attention_forward(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        ):
            num_att_heads = vlm_with_expert.num_attention_heads
            num_key_value_heads = vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            
            sequence_length = key_states.shape[1]
            
            key_states_expanded = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            ).reshape(batch_size, sequence_length, num_att_heads, head_dim)
            
            value_states_expanded = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            ).reshape(batch_size, sequence_length, num_att_heads, head_dim)
            
            query_states_t = query_states.transpose(1, 2).float()
            key_states_t = key_states_expanded.transpose(1, 2).float()
            
            att_weights = torch.matmul(query_states_t, key_states_t.transpose(2, 3))
            att_weights *= head_dim ** -0.5
            
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
            
            layer_idx = len(attention_dict)
            attention_dict[f"layer_{layer_idx}"] = probs.detach().cpu()
            
            probs = probs.to(dtype=value_states_expanded.dtype)
            att_output = torch.matmul(probs, value_states_expanded.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(batch_size, -1, num_att_heads * head_dim)
            
            return att_output
        
        images, img_masks, lang_tokens, lang_masks, state_tensor = self._prepare_inputs(
            image, instruction, state
        )
        
        vlm_with_expert.eager_attention_forward = patched_attention_forward
        
        try:
            with torch.no_grad():
                actions = self.policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state_tensor
                )
        finally:
            vlm_with_expert.eager_attention_forward = original_eager_attention
        
        self.attention_cache = attention_dict
        return attention_dict, actions
    
    def _extract_image_attention(
        self,
        attention: torch.Tensor,
        num_image_tokens: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract attention weights for image tokens.
        
        Args:
            attention: Attention weights [batch, heads, seq_q, seq_k] or [heads, seq_q, seq_k]
            num_image_tokens: Number of image tokens
        
        Returns:
            Image attention heatmap [H, W], normalized to [0, 1]
        """
        if num_image_tokens is None:
            num_image_tokens = self.image_token_info.get('num_image_tokens', 729)
        
        num_patches_per_side = self.image_token_info.get('num_patches_per_side', 27)
        
        if len(attention.shape) == 4:
            attn = attention[0]  # [heads, seq_q, seq_k]
        else:
            attn = attention  # [heads, seq_q, seq_k]
        
        attn_avg = attn.mean(0).float().numpy()  # [seq_q, seq_k]
        
        seq_q, seq_k = attn_avg.shape
        
        if seq_k >= num_image_tokens:
            image_attn = attn_avg[:, :num_image_tokens]  # [seq_q, num_image_tokens]
            image_attn_avg = image_attn.mean(0)  # [num_image_tokens]
        else:
            image_attn_avg = attn_avg.mean(0)  # [seq_k]
            num_patches_per_side = int(np.sqrt(len(image_attn_avg)))
            num_image_tokens = num_patches_per_side ** 2
            image_attn_avg = image_attn_avg[:num_image_tokens]
        
        try:
            actual_tokens = len(image_attn_avg)
            side = int(np.sqrt(actual_tokens))
            if side * side != actual_tokens:
                side = int(np.ceil(np.sqrt(actual_tokens)))
                padded = np.zeros(side * side)
                padded[:actual_tokens] = image_attn_avg
                image_attn_avg = padded
            
            attn_map = image_attn_avg.reshape(side, side)
        except Exception as e:
            print(f"Warning: Could not reshape attention to 2D grid: {e}")
            attn_map = np.zeros((num_patches_per_side, num_patches_per_side))
            flat_len = min(len(image_attn_avg), num_patches_per_side * num_patches_per_side)
            attn_map.flat[:flat_len] = image_attn_avg[:flat_len]
        
        attn_min = attn_map.min()
        attn_max = attn_map.max()
        if attn_max > attn_min:
            attn_map = (attn_map - attn_min) / (attn_max - attn_min)
        else:
            attn_map = np.zeros_like(attn_map)
        
        return attn_map
    
    def _create_attention_overlay(
        self,
        image: Image.Image,
        attention_map: np.ndarray,
        alpha: float = 0.5,
        cmap: str = "jet"
    ) -> np.ndarray:
        """
        Overlay attention heatmap on the original image.
        
        Args:
            image: Original PIL image
            attention_map: 2D attention heatmap [H, W], values in [0, 1]
            alpha: Heatmap transparency
            cmap: Matplotlib colormap name
        
        Returns:
            Blended image numpy array [H, W, 3]
        """
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        attn_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        colormap = plt.get_cmap(cmap)
        attn_colored = colormap(attn_resized)[:, :, :3]  # [H, W, 3], values in [0, 1]
        attn_colored = (attn_colored * 255).astype(np.uint8)
        
        img_float = img_np.astype(np.float32)
        attn_float = attn_colored.astype(np.float32)
        
        blended = (1 - alpha) * img_float + alpha * attn_float
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended
    
    def visualize_attention_on_image(
        self,
        attention_weights: Dict[str, torch.Tensor],
        image: Image.Image,
        instruction: str,
        selected_layers: Optional[List[int]] = None,
        prefix: str = "attention_overlay"
    ):
        """
        Create attention overlay visualization.
        
        Args:
            attention_weights: Attention weights for each layer
            image: Original input image
            instruction: Task instruction
            selected_layers: List of layer indices to visualize, None to auto-select
            prefix: Output file name prefix
        """
        if not attention_weights:
            print("No attention weights to visualize")
            return
        
        layers = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[-1]))
        n_layers = len(layers)
        
        if selected_layers is None:
            if n_layers >= 4:
                indices = [
                    n_layers // 6,
                    n_layers // 3,
                    n_layers // 2,
                    2 * n_layers // 3,
                ]
            else:
                indices = list(range(n_layers))
            selected_layers = indices
        
        selected_layers = [i for i in selected_layers if i < n_layers]
        n_selected = len(selected_layers)
        
        if n_selected == 0:
            print("No valid layers to visualize")
            return
        
        n_cols = min(4, n_selected)
        n_rows = (n_selected + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f"Attention Overlay on Image\nInstruction: {instruction[:50]}...", fontsize=14, y=1.02)
        
        for idx, layer_idx in enumerate(selected_layers):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            layer_name = layers[layer_idx]
            attn = attention_weights[layer_name]
            
            attn_map = self._extract_image_attention(attn)
            
            overlay = self._create_attention_overlay(
                image, 
                attn_map,
                alpha=self.config.overlay_alpha,
                cmap=self.config.overlay_cmap
            )
            
            ax.imshow(overlay)
            ax.set_title(f"Layer {layer_idx}", fontsize=12)
            ax.axis('off')
        
        for idx in range(n_selected, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.current_output_dir, f"{prefix}_grid.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        
        self._save_individual_overlays(attention_weights, image, selected_layers, layers, prefix)
    
    def _save_individual_overlays(
        self,
        attention_weights: Dict[str, torch.Tensor],
        image: Image.Image,
        selected_layers: List[int],
        layers: List[str],
        prefix: str
    ):
        """Save individual attention overlay images"""
        overlay_dir = os.path.join(self.current_output_dir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        
        for layer_idx in selected_layers:
            layer_name = layers[layer_idx]
            attn = attention_weights[layer_name]
            
            attn_map = self._extract_image_attention(attn)
            overlay = self._create_attention_overlay(
                image,
                attn_map,
                alpha=self.config.overlay_alpha,
                cmap=self.config.overlay_cmap
            )
            
            save_path = os.path.join(overlay_dir, f"{prefix}_layer_{layer_idx}.png")
            Image.fromarray(overlay).save(save_path)
        
        print(f"Saved individual overlays to: {overlay_dir}")
    
    def create_comparison_visualization(
        self,
        attention_weights: Dict[str, torch.Tensor],
        image: Image.Image,
        instruction: str,
        layer_indices: Optional[List[int]] = None,
        prefix: str = "comparison"
    ):
        """
        Create comparison visualization similar to OpenVLA paper.
        
        Args:
            attention_weights: Attention weights for each layer
            image: Original input image
            instruction: Task instruction
            layer_indices: Layers to display
            prefix: Output file name prefix
        """
        layers = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[-1]))
        n_layers = len(layers)
        
        if layer_indices is None:
            mid = n_layers // 2
            layer_indices = list(range(max(0, mid - 2), min(n_layers, mid + 2)))
        
        n_display = len(layer_indices)
        
        fig = plt.figure(figsize=(4 * (n_display + 1), 5))
        
        ax_img = fig.add_subplot(1, n_display + 1, 1)
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12)
        ax_img.axis('off')
        
        wrapped_instruction = '\n'.join([instruction[i:i+30] for i in range(0, len(instruction), 30)])
        ax_img.text(0.5, -0.1, f"Instruction:\n{wrapped_instruction}", 
                    transform=ax_img.transAxes, fontsize=9,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        for i, layer_idx in enumerate(layer_indices):
            ax = fig.add_subplot(1, n_display + 1, i + 2)
            
            if layer_idx < n_layers:
                layer_name = layers[layer_idx]
                attn = attention_weights[layer_name]
                
                attn_map = self._extract_image_attention(attn)
                overlay = self._create_attention_overlay(
                    image,
                    attn_map,
                    alpha=self.config.overlay_alpha,
                    cmap=self.config.overlay_cmap
                )
                
                ax.imshow(overlay)
                ax.set_title(f"Layer {layer_idx}", fontsize=12)
            
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.current_output_dir, f"{prefix}_paper_style.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    
    def visualize_all_layers(
        self,
        attention_weights: Dict[str, torch.Tensor],
        image: Image.Image,
        instruction: str,
        prefix: str = "attention"
    ):
        """Visualize attention heatmaps for all layers"""
        if not attention_weights:
            print("No attention weights captured.")
            return
        
        print(f"Saving outputs to: {self.current_output_dir}")
        
        self._create_layer_grid(attention_weights, prefix)
        self._create_overview(attention_weights, image, instruction, prefix)
        self.visualize_attention_on_image(
            attention_weights, image, instruction, prefix=f"{prefix}_overlay"
        )
        self.create_comparison_visualization(
            attention_weights, image, instruction, prefix=f"{prefix}_comparison"
        )
        
        if self.config.save_individual:
            for layer_name, attn in attention_weights.items():
                self._save_single_heatmap(attn, layer_name, prefix)
    
    def _create_layer_grid(self, attention_weights: Dict[str, torch.Tensor], prefix: str):
        """Create grid of all layer attention maps"""
        layers = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[-1]))
        n_layers = len(layers)
        
        n_cols = min(6, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle("SmolVLA Attention Across Layers", fontsize=14, y=1.02)
        
        for idx, layer_name in enumerate(layers):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            attn = attention_weights[layer_name]
            if len(attn.shape) == 4:
                attn_2d = attn[0].mean(0).float().numpy()
            else:
                attn_2d = attn.mean(0).float().numpy()
            
            ax.imshow(attn_2d, cmap='magma', aspect='auto')
            ax.set_title(f"Layer {layer_name.split('_')[-1]}", fontsize=9)
            ax.tick_params(labelsize=5)
        
        for idx in range(n_layers, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.current_output_dir, f"{prefix}_layer_grid.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    
    def _create_overview(
        self,
        attention_weights: Dict[str, torch.Tensor],
        image: Image.Image,
        instruction: str,
        prefix: str
    ):
        """Create overview visualization with image and attention"""
        fig = plt.figure(figsize=self.config.figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12)
        ax_img.axis('off')
        
        ax_text = fig.add_subplot(gs[1, 0])
        ax_text.text(0.5, 0.5, f"Instruction:\n{instruction}",
                     ha='center', va='center', fontsize=10, wrap=True,
                     transform=ax_text.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_text.axis('off')
        
        layers = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[-1]))
        n_display = min(9, len(layers))
        step = max(1, len(layers) // n_display)
        selected = layers[::step][:n_display]
        
        for idx, layer_name in enumerate(selected):
            row, col = idx // 3, idx % 3 + 1
            ax = fig.add_subplot(gs[row, col])
            
            attn = attention_weights[layer_name]
            if len(attn.shape) == 4:
                attn_viz = attn[0].mean(0).float().numpy()
            else:
                attn_viz = attn.mean(0).float().numpy()
            
            ax.imshow(attn_viz, cmap=self.config.cmap, aspect='auto')
            ax.set_title(f"Layer {layer_name.split('_')[-1]}", fontsize=9)
            ax.tick_params(labelsize=6)
        
        fig.suptitle("SmolVLA Attention Visualization", fontsize=16, y=0.98)
        
        save_path = os.path.join(self.current_output_dir, f"{prefix}_overview.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    
    def _save_single_heatmap(self, attn: torch.Tensor, layer_name: str, prefix: str):
        """Save individual layer heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(attn.shape) == 4:
            attn_2d = attn[0].mean(0).float().numpy()
        else:
            attn_2d = attn.mean(0).float().numpy()
        
        im = ax.imshow(attn_2d, cmap=self.config.cmap, aspect='auto')
        ax.set_title(f"Attention: {layer_name}", fontsize=14)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        
        save_path = os.path.join(self.current_output_dir, f"{prefix}_{layer_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def cleanup(self):
        """Clean up resources"""
        if self.policy is not None:
            del self.policy
        torch.cuda.empty_cache()


def main():
    config = AttentionConfig(
        model_path="lerobot/smolvla_base",
        input_dir="./myscripts/attention/inputs",
        output_dir="./myscripts/attention/outputs",
        overlay_alpha=0.5,
        overlay_cmap="jet",
    )
    
    visualizer = SmolVLAAttentionVisualizer(config)
    
    try:
        visualizer.load_model()
        
        input_images = list(Path(config.input_dir).glob("*.png")) + \
                       list(Path(config.input_dir).glob("*.jpg")) + \
                       list(Path(config.input_dir).glob("*.jpeg"))
        
        if input_images:
            image_path = str(input_images[0])
            image = Image.open(image_path).convert('RGB')
            image_name = Path(image_path).stem
            print(f"Processing image: {image_path}")
        else:
            image = Image.new('RGB', (640, 480), color=(100, 150, 200))
            image_name = "test_image"
            print("No images in input folder, using test image")
        
        instruction = "Pick up the red block and place it on the blue plate."
        
        attention_weights, actions = visualizer.extract_attention_with_output(
            image, instruction, image_name=image_name
        )
        
        print(f"Captured attention from {len(attention_weights)} layers")
        visualizer.visualize_all_layers(attention_weights, image, instruction)
        
    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    main()