"""
SmolVLA Attention Visualization

核心功能:
1. 提取 VLM 自注意力和 Denoising 交叉注意力
2. 可视化注意力热图叠加到原图
3. 分析跨相机、跨模态的注意力分布
4. 计算注意力统计数据
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

sys.path.insert(0, "/home/zwt/Projects/lerobot")
sys.path.insert(0, "/home/zwt/Projects/lerobot/src")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@dataclass
class AttentionConfig:
    """注意力可视化配置"""
    model_path: str = "lerobot/smolvla_base"
    input_dir: str = "./myscripts/attention/inputs"
    output_dir: str = "./myscripts/attention/outputs"
    overlay_alpha: float = 0.5
    overlay_cmap: str = "jet"
    save_raw_attention_matrix: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SequenceLayout:
    """序列布局信息"""
    image_token_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    text_token_range: Tuple[int, int] = (0, 0)
    state_token_range: Tuple[int, int] = (0, 0)
    tokens_per_image: int = 64
    patches_per_side: int = 8
    total_length: int = 0
    camera_order: List[str] = field(default_factory=list)


class SmolVLAAttentionVisualizer:
    """SmolVLA 注意力可视化器"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.policy = None
        self.processor = None
        self.normalization_stats = None
        self.sequence_layout: Optional[SequenceLayout] = None
        self.camera_keys: List[str] = []
        self.current_output_dir = config.output_dir
        self.tokens_per_image = 64
        self.patches_per_side = 8
        self.vlm_attention_cache: Dict[str, torch.Tensor] = {}
        self.denoising_attention_cache: Dict[str, torch.Tensor] = {}
        
        os.makedirs(config.input_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_model(self):
        """加载模型"""
        print(f"Loading SmolVLA from {self.config.model_path}...")
        self.policy = SmolVLAPolicy.from_pretrained(self.config.model_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.processor = self.policy.model.vlm_with_expert.processor
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        vlm = self.policy.model.vlm_with_expert
        print("\n" + "=" * 50)
        print("SmolVLA Model Info:")
        print(f"  VLM layers: {vlm.num_vlm_layers}")
        print(f"  Expert layers: {vlm.num_expert_layers}")
        print(f"  Num attention heads: {vlm.num_attention_heads}")
        print(f"  Num KV heads: {vlm.num_key_value_heads}")
        print("=" * 50 + "\n")
    
    def set_normalization_stats(self, stats: Dict):
        """设置归一化统计量"""
        self.normalization_stats = stats
    
    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """归一化状态"""
        if self.normalization_stats is None:
            return state
        
        state_key = "observation.state"
        if state_key not in self.normalization_stats:
            return state
        
        stats = self.normalization_stats[state_key]
        mean = stats.get("mean")
        std = stats.get("std")
        
        if mean is None or std is None:
            return state
        
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        elif not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()
        elif not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)
        
        mean = mean.to(state.device)
        std = std.to(state.device)
        
        state_dim = state.shape[-1]
        mean_dim = mean.shape[-1] if mean.dim() > 0 else 1
        
        if mean_dim > state_dim:
            mean = mean[..., :state_dim]
            std = std[..., :state_dim]
        elif mean_dim < state_dim:
            pad_size = state_dim - mean_dim
            mean = F.pad(mean, (0, pad_size), value=0)
            std = F.pad(std, (0, pad_size), value=1)
        
        std = torch.clamp(std, min=1e-8)
        return (state - mean) / std
    
    def _prepare_inputs_like_inference(
        self,
        images: Dict[str, Image.Image],
        instruction: str,
        state: Optional[torch.Tensor] = None,
        normalize_state: bool = True
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备输入 - 与推理流程一致"""
        self.camera_keys = list(images.keys())
        
        processed_images = []
        img_masks = []
        target_size = self.policy.config.resize_imgs_with_padding
        
        for key in self.camera_keys:
            img = images[key]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            if target_size is not None:
                img_tensor = self._resize_with_pad(img_tensor, target_size[1], target_size[0], pad_value=-1)
            
            img_tensor = img_tensor * 2.0 - 1.0
            img_tensor = img_tensor.to(self.device)
            processed_images.append(img_tensor)
            img_masks.append(torch.ones(1, dtype=torch.bool, device=self.device))
        
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
            state_tensor = torch.zeros(1, self.policy.config.max_state_dim, device=self.device)
        else:
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float()
            else:
                state_tensor = state.float()
            
            state_tensor = state_tensor.to(self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            if normalize_state:
                state_tensor = self._normalize_state(state_tensor)
            
            if state_tensor.shape[-1] < self.policy.config.max_state_dim:
                padding = torch.zeros(
                    state_tensor.shape[0],
                    self.policy.config.max_state_dim - state_tensor.shape[-1],
                    device=self.device
                )
                state_tensor = torch.cat([state_tensor, padding], dim=-1)
        
        return processed_images, img_masks, lang_tokens, lang_masks, state_tensor
    
    def _resize_with_pad(self, img: torch.Tensor, width: int, height: int, pad_value: float = -1) -> torch.Tensor:
        """Resize with padding"""
        cur_height, cur_width = img.shape[2:]
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        
        resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        
        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))
        
        return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    
    def _compute_sequence_layout(
        self,
        attention_shape: Tuple[int, ...],
        num_cameras: int,
        lang_token_count: int
    ) -> SequenceLayout:
        """计算序列布局"""
        layout = SequenceLayout()
        layout.camera_order = self.camera_keys.copy()
        layout.tokens_per_image = self.tokens_per_image
        layout.patches_per_side = self.patches_per_side
        
        if len(attention_shape) == 4:
            seq_len = attention_shape[2]
        elif len(attention_shape) == 3:
            seq_len = attention_shape[1]
        else:
            seq_len = attention_shape[0]
        
        layout.total_length = seq_len
        current_pos = 0
        
        for cam_key in self.camera_keys:
            start = current_pos
            end = current_pos + self.tokens_per_image
            if end > seq_len:
                end = min(end, seq_len)
            layout.image_token_ranges[cam_key] = (start, end)
            current_pos = end
        
        estimated_text_tokens = seq_len - (num_cameras * self.tokens_per_image) - 1
        if lang_token_count > 0 and lang_token_count <= estimated_text_tokens + 10:
            text_tokens = lang_token_count
        else:
            text_tokens = max(0, estimated_text_tokens)
        
        text_start = current_pos
        text_end = min(current_pos + text_tokens, seq_len - 1)
        layout.text_token_range = (text_start, text_end)
        current_pos = text_end
        
        if current_pos < seq_len:
            layout.state_token_range = (current_pos, seq_len)
        else:
            layout.state_token_range = (seq_len - 1, seq_len)
        
        print(f"\nSequence Layout (total={seq_len}):")
        for cam_key, (s, e) in layout.image_token_ranges.items():
            print(f"  {cam_key.split('.')[-1]}: [{s}, {e}) = {e-s} tokens")
        print(f"  text: [{layout.text_token_range[0]}, {layout.text_token_range[1]})")
        print(f"  state: [{layout.state_token_range[0]}, {layout.state_token_range[1]})")
        
        self.sequence_layout = layout
        return layout
    
    def extract_attention_with_output(
        self,
        images: Dict[str, Image.Image],
        instruction: str,
        state: Optional[torch.Tensor] = None,
        image_name: Optional[str] = None,
        normalize_state: bool = True
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        """提取注意力权重和模型输出"""
        if image_name:
            self.current_output_dir = os.path.join(self.config.output_dir, image_name)
            os.makedirs(self.current_output_dir, exist_ok=True)
        
        self.vlm_attention_cache = {}
        self.denoising_attention_cache = {}
        
        vlm_with_expert = self.policy.model.vlm_with_expert
        num_vlm_layers = vlm_with_expert.num_vlm_layers
        original_attention_forward = vlm_with_expert.eager_attention_forward
        
        attention_counter = [0]
        is_prefix_phase = [True]
        
        def patched_attention_forward(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        ):
            num_att_heads = vlm_with_expert.num_attention_heads
            num_key_value_heads = vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            
            seq_len_q = query_states.shape[1]
            seq_len_k = key_states.shape[1]
            
            current_is_prefix = (seq_len_q == seq_len_k)
            
            if not current_is_prefix and is_prefix_phase[0]:
                is_prefix_phase[0] = False
                attention_counter[0] = 0
            
            key_states_expanded = key_states[:, :, :, None, :].expand(
                batch_size, seq_len_k, num_key_value_heads, num_key_value_groups, head_dim
            ).reshape(batch_size, seq_len_k, num_att_heads, head_dim)
            
            value_states_expanded = value_states[:, :, :, None, :].expand(
                batch_size, seq_len_k, num_key_value_heads, num_key_value_groups, head_dim
            ).reshape(batch_size, seq_len_k, num_att_heads, head_dim)
            
            query_f32 = query_states.to(torch.float32).transpose(1, 2)
            key_f32 = key_states_expanded.to(torch.float32).transpose(1, 2)
            
            attn_weights = torch.matmul(query_f32, key_f32.transpose(-2, -1))
            attn_weights *= head_dim ** -0.5
            
            if attention_mask is not None:
                big_neg = torch.finfo(attn_weights.dtype).min
                if attention_mask.dim() == 2:
                    mask = attention_mask[:, None, None, :]
                elif attention_mask.dim() == 3:
                    mask = attention_mask[:, None, :, :]
                else:
                    mask = attention_mask
                
                if mask.shape[-1] != attn_weights.shape[-1]:
                    if mask.shape[-1] > attn_weights.shape[-1]:
                        mask = mask[..., :attn_weights.shape[-1]]
                    else:
                        pad_size = attn_weights.shape[-1] - mask.shape[-1]
                        mask = F.pad(mask.float(), (0, pad_size), value=0).bool()
                
                masked_weights = torch.where(mask, attn_weights, big_neg)
            else:
                masked_weights = attn_weights
            
            attn_probs = F.softmax(masked_weights, dim=-1)
            
            layer_idx = attention_counter[0] % num_vlm_layers
            
            if is_prefix_phase[0]:
                key_name = f"layer_{layer_idx}"
                self.vlm_attention_cache[key_name] = attn_probs.detach().cpu()
            else:
                step_num = attention_counter[0] // num_vlm_layers
                key_name = f"step_{step_num}_layer_{layer_idx}"
                self.denoising_attention_cache[key_name] = attn_probs.detach().cpu()
            
            attention_counter[0] += 1
            
            attn_probs_v = attn_probs.to(value_states_expanded.dtype)
            attn_output = torch.matmul(attn_probs_v, value_states_expanded.transpose(1, 2))
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(batch_size, -1, num_att_heads * head_dim)
            
            return attn_output
        
        processed_images, img_masks, lang_tokens, lang_masks, state_tensor = \
            self._prepare_inputs_like_inference(images, instruction, state, normalize_state)
        
        vlm_with_expert.eager_attention_forward = patched_attention_forward
        
        try:
            with torch.no_grad():
                actions = self.policy.model.sample_actions(
                    processed_images, img_masks, lang_tokens, lang_masks, state_tensor
                )
        finally:
            vlm_with_expert.eager_attention_forward = original_attention_forward
        
        print(f"\nAttention Capture Summary:")
        print(f"  VLM layers captured: {len(self.vlm_attention_cache)}")
        print(f"  Denoising entries captured: {len(self.denoising_attention_cache)}")
        
        if self.vlm_attention_cache:
            first_key = list(self.vlm_attention_cache.keys())[0]
            first_attn = self.vlm_attention_cache[first_key]
            print(f"  VLM attention shape: {first_attn.shape}")
            lang_token_count = lang_masks.sum().item()
            self._compute_sequence_layout(first_attn.shape, len(self.camera_keys), int(lang_token_count))
        
        if self.denoising_attention_cache:
            first_key = list(self.denoising_attention_cache.keys())[0]
            first_attn = self.denoising_attention_cache[first_key]
            print(f"  Denoising attention shape: {first_attn.shape}")
        
        attention_data = {
            "vlm": self.vlm_attention_cache,
            "denoising": self.denoising_attention_cache
        }
        
        return attention_data, actions
    
    def _extract_camera_attention_from_vlm(
        self,
        attention: torch.Tensor,
        camera_key: str,
        query_type: str = "all"
    ) -> np.ndarray:
        """从 VLM 自注意力中提取特定相机的注意力
        
        注意力矩阵 attn[i, j] 表示位置 i (Query) 对位置 j (Key) 的注意力
        
        Args:
            attention: 注意力权重 [batch, heads, seq_q, seq_k]
            camera_key: 目标相机名称
            query_type: 
                - "to_text": 该图像关注文本的哪些部分（Image→Text）
                - "to_other_images": 该图像关注其他图像的哪些部分
                - "self": 该图像内部的自注意力模式
                - "from_text": 文本关注该图像的哪些部分（Text→Image）
                - "from_all": 所有其他 token 关注该图像的哪些部分
        """
        if self.sequence_layout is None:
            raise ValueError("Sequence layout not computed")
        
        layout = self.sequence_layout
        if camera_key not in layout.image_token_ranges:
            raise ValueError(f"Camera {camera_key} not found")
        
        img_start, img_end = layout.image_token_ranges[camera_key]
        
        if attention.dim() == 4:
            attn = attention[0]
        else:
            attn = attention
        
        attn_avg = attn.mean(0).float().numpy()  # [seq_q, seq_k]
        seq_len = attn_avg.shape[0]
        
        if query_type == "from_text":
            # 文本 (Query) → 图像 (Key): 文本关注图像的哪些区域
            # attn[text_positions, image_positions]
            q_start, q_end = layout.text_token_range
            image_attn = attn_avg[q_start:q_end, img_start:img_end]
            # 沿 Query 维度求和，得到每个图像 token 被关注的程度
            attn_per_patch = image_attn.sum(axis=0)
            
        elif query_type == "self":
            # 该图像内部的自注意力
            # attn[image_positions, image_positions]
            self_attn = attn_avg[img_start:img_end, img_start:img_end]
            # 取列的和：每个 patch 被其他 patch 关注的程度
            attn_per_patch = self_attn.sum(axis=0)
            
        elif query_type == "from_other_images":
            # 其他图像 (Query) → 该图像 (Key)
            attn_per_patch = np.zeros(img_end - img_start)
            for other_cam, (other_start, other_end) in layout.image_token_ranges.items():
                if other_cam == camera_key:
                    continue
                other_to_this = attn_avg[other_start:other_end, img_start:img_end]
                attn_per_patch += other_to_this.sum(axis=0)
                
        elif query_type == "from_all":
            # 所有其他 token → 该图像（排除该图像自身作为 Query）
            attn_per_patch = np.zeros(img_end - img_start)
            
            # 该图像之前的所有 token
            if img_start > 0:
                before_attn = attn_avg[0:img_start, img_start:img_end]
                attn_per_patch += before_attn.sum(axis=0)
            
            # 该图像之后的所有 token
            if img_end < seq_len:
                after_attn = attn_avg[img_end:seq_len, img_start:img_end]
                attn_per_patch += after_attn.sum(axis=0)
        
        elif query_type == "to_text":
            # 该图像 (Query) → 文本 (Key): 图像在关注哪些文本
            # 这个返回的是文本注意力，不是图像热图
            q_start, q_end = layout.text_token_range
            image_to_text = attn_avg[img_start:img_end, q_start:q_end]
            # 这里返回的是 [64, text_len] 的矩阵，需要特殊处理
            attn_per_patch = image_to_text.mean(axis=1)  # 每个图像 patch 对文本的平均注意力
        
        else:
            raise ValueError(f"Unknown query_type: {query_type}")
        
        # 重塑为 2D
        side = layout.patches_per_side
        if len(attn_per_patch) == side * side:
            attn_2d = attn_per_patch.reshape(side, side)
        else:
            attn_2d = self._reshape_to_2d(attn_per_patch, side)
        
        return self._normalize_attention(attn_2d)
    
    def _extract_camera_attention_from_denoising(
        self,
        attention: torch.Tensor,
        camera_key: str
    ) -> np.ndarray:
        """从 Denoising 交叉注意力中提取特定相机的注意力"""
        if self.sequence_layout is None:
            raise ValueError("Sequence layout not computed")
        
        layout = self.sequence_layout
        if camera_key not in layout.image_token_ranges:
            raise ValueError(f"Camera {camera_key} not found")
        
        img_start, img_end = layout.image_token_ranges[camera_key]
        
        if attention.dim() == 4:
            attn = attention[0]
        else:
            attn = attention
        
        attn_avg = attn.mean(0).float().numpy()
        seq_q, seq_k = attn_avg.shape
        
        if seq_k >= img_end:
            image_attn = attn_avg[:, img_start:img_end]
        else:
            cam_idx = list(layout.image_token_ranges.keys()).index(camera_key)
            local_start = cam_idx * layout.tokens_per_image
            local_end = min(local_start + layout.tokens_per_image, seq_k)
            image_attn = attn_avg[:, local_start:local_end]
        
        attn_mean = image_attn.mean(axis=0)
        
        side = layout.patches_per_side
        if len(attn_mean) == side * side:
            attn_2d = attn_mean.reshape(side, side)
        else:
            attn_2d = self._reshape_to_2d(attn_mean, side)
        
        return self._normalize_attention(attn_2d)
    
    def _reshape_to_2d(self, attn_1d: np.ndarray, target_side: int) -> np.ndarray:
        """将 1D 注意力重塑为 2D"""
        actual_len = len(attn_1d)
        target_len = target_side * target_side
        
        if actual_len == target_len:
            return attn_1d.reshape(target_side, target_side)
        
        actual_side = int(np.sqrt(actual_len))
        if actual_side * actual_side == actual_len:
            attn_2d = attn_1d.reshape(actual_side, actual_side)
            if actual_side != target_side:
                attn_2d = cv2.resize(attn_2d.astype(np.float32), 
                                     (target_side, target_side),
                                     interpolation=cv2.INTER_LINEAR)
            return attn_2d
        
        padded = np.zeros(target_len)
        padded[:min(actual_len, target_len)] = attn_1d[:min(actual_len, target_len)]
        return padded.reshape(target_side, target_side)
    
    def _normalize_attention(self, attn: np.ndarray) -> np.ndarray:
        """归一化注意力到 [0, 1]"""
        attn_min, attn_max = attn.min(), attn.max()
        if attn_max > attn_min:
            return (attn - attn_min) / (attn_max - attn_min)
        return np.zeros_like(attn)
    
    def _create_overlay(
        self,
        image: Image.Image,
        attention_map: np.ndarray,
        alpha: float = 0.5,
        cmap: str = "jet"
    ) -> np.ndarray:
        """创建注意力叠加图 - 考虑 padding 偏移"""
        img_np = np.array(image)
        orig_h, orig_w = img_np.shape[:2]  # 640×480 或 480×640
        
        target_size = self.policy.config.resize_imgs_with_padding
        if target_size is not None:
            target_h, target_w = target_size  # (512, 512)
            
            # 计算缩放比例和 padding
            ratio = max(orig_w / target_w, orig_h / target_h)
            resized_w = int(orig_w / ratio)
            resized_h = int(orig_h / ratio)
            
            pad_top = target_h - resized_h  # 填充在顶部
            pad_left = target_w - resized_w  # 填充在左边
            
            # 计算 padding 对应的 patch 数量
            patches_per_side = self.patches_per_side  # 8
            patch_size = target_h // patches_per_side  # 512/8 = 64
            
            pad_top_patches = pad_top // patch_size
            pad_left_patches = pad_left // patch_size
            
            # 从注意力图中提取实际图像区域
            valid_attn = attention_map[pad_top_patches:, pad_left_patches:]
            
            # 如果有效区域为空，使用原始注意力图
            if valid_attn.size == 0:
                valid_attn = attention_map
        else:
            valid_attn = attention_map
        
        # 上采样到原始图像尺寸
        attn_resized = cv2.resize(
            valid_attn.astype(np.float32), 
            (orig_w, orig_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 应用颜色映射
        colormap = plt.get_cmap(cmap)
        attn_colored = colormap(attn_resized)[:, :, :3]
        attn_colored = (attn_colored * 255).astype(np.uint8)
        
        # 混合
        blended = ((1 - alpha) * img_np + alpha * attn_colored).astype(np.uint8)
        return blended
    
    def visualize_vlm_attention(
        self,
        attention_dict: Dict[str, torch.Tensor],
        images: Dict[str, Image.Image],
        save_prefix: str = "vlm"
    ):
        """可视化 VLM 层的注意力 - 聚焦最有价值的分析"""
        if not attention_dict:
            print("No VLM attention to visualize")
            return
        
        vlm_dir = os.path.join(self.current_output_dir, "vlm_attention")
        os.makedirs(vlm_dir, exist_ok=True)
        
        layer_names = sorted(attention_dict.keys(), key=lambda x: int(x.split('_')[-1]))
        # 只分析最后一层（特征最成熟）
        last_layer_name = layer_names[-1]
        attn = attention_dict[last_layer_name]
        
        layout = self.sequence_layout
        if layout is None:
            print("Sequence layout not computed")
            return
        
        if attn.dim() == 4:
            attn_tensor = attn[0]
        else:
            attn_tensor = attn
        
        attn_avg = attn_tensor.mean(0).float().numpy()  # [seq_q, seq_k]
        
        # ========== 1. Text → Image (Language Grounding) ==========
        text_to_image_dir = os.path.join(vlm_dir, "text_to_image")
        os.makedirs(text_to_image_dir, exist_ok=True)
        
        text_start, text_end = layout.text_token_range
        
        for cam_key, image in images.items():
            img_start, img_end = layout.image_token_ranges[cam_key]
            
            # Text tokens 对 Image tokens 的注意力
            text_to_img = attn_avg[text_start:text_end, img_start:img_end]
            attn_per_patch = text_to_img.sum(axis=0)  # 每个 patch 被文本关注的程度
            
            attn_2d = self._reshape_to_2d(attn_per_patch, layout.patches_per_side)
            attn_2d = self._normalize_attention(attn_2d)
            
            overlay = self._create_overlay(image, attn_2d, self.config.overlay_alpha)
            
            # 保存
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(np.array(image))
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            im = axes[1].imshow(attn_2d, cmap='jet')
            axes[1].set_title("Attention Heatmap\n(Text → Image)")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            
            cam_short = cam_key.split('.')[-1]
            plt.suptitle(f"Language Grounding: Where does the instruction look? ({cam_short})")
            plt.tight_layout()
            
            save_path = os.path.join(text_to_image_dir, f"{cam_short}_grounding.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # ========== 2. Cross-Camera Attention ==========
        if len(images) > 1:
            cross_cam_dir = os.path.join(vlm_dir, "cross_camera")
            os.makedirs(cross_cam_dir, exist_ok=True)
            
            camera_keys = list(images.keys())
            n_cams = len(camera_keys)
            
            # 创建详细的跨相机分析
            fig, axes = plt.subplots(n_cams, n_cams, figsize=(4*n_cams, 4*n_cams))
            
            for i, src_cam in enumerate(camera_keys):
                src_start, src_end = layout.image_token_ranges[src_cam]
                
                for j, tgt_cam in enumerate(camera_keys):
                    tgt_start, tgt_end = layout.image_token_ranges[tgt_cam]
                    
                    ax = axes[i, j] if n_cams > 1 else axes
                    
                    # src 关注 tgt 的哪里
                    cross_attn = attn_avg[src_start:src_end, tgt_start:tgt_end]
                    attn_per_patch = cross_attn.sum(axis=0)
                    
                    attn_2d = self._reshape_to_2d(attn_per_patch, layout.patches_per_side)
                    attn_2d = self._normalize_attention(attn_2d)
                    
                    # 叠加到目标图像
                    tgt_image = images[tgt_cam]
                    overlay = self._create_overlay(tgt_image, attn_2d, 0.6)
                    
                    ax.imshow(overlay)
                    src_short = src_cam.split('.')[-1]
                    tgt_short = tgt_cam.split('.')[-1]
                    
                    if i == j:
                        ax.set_title(f"{src_short}\n(Self-Attention)", fontsize=10)
                    else:
                        ax.set_title(f"{src_short} → {tgt_short}", fontsize=10)
                    ax.axis('off')
            
            plt.suptitle("Cross-Camera Attention: Which camera looks at which regions?", fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(cross_cam_dir, "cross_camera_detailed.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"VLM attention saved to {vlm_dir}")

    def visualize_denoising_attention(
        self,
        attention_dict: Dict[str, torch.Tensor],
        images: Dict[str, Image.Image],
        save_prefix: str = "denoising"
    ):
        """可视化 Denoising 层的交叉注意力"""
        if not attention_dict:
            print("No denoising attention to visualize")
            return
        
        denoising_dir = os.path.join(self.current_output_dir, "denoising_attention")
        os.makedirs(denoising_dir, exist_ok=True)
        
        steps = {}
        for key, attn in attention_dict.items():
            parts = key.split('_')
            step_num = int(parts[1])
            if step_num not in steps:
                steps[step_num] = {}
            steps[step_num][key] = attn
        
        for step_num in sorted(steps.keys()):
            step_dir = os.path.join(denoising_dir, f"step_{step_num}")
            os.makedirs(step_dir, exist_ok=True)
            
            step_layers = steps[step_num]
            last_layer_key = max(step_layers.keys(), key=lambda x: int(x.split('_')[-1]))
            attn = step_layers[last_layer_key]
            
            for cam_key, image in images.items():
                try:
                    attn_map = self._extract_camera_attention_from_denoising(attn, cam_key)
                    overlay = self._create_overlay(image, attn_map, self.config.overlay_alpha)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    
                    axes[0].imshow(np.array(image))
                    axes[0].set_title("Original")
                    axes[0].axis('off')
                    
                    axes[1].imshow(attn_map, cmap='jet')
                    axes[1].set_title("Attention Map")
                    axes[1].axis('off')
                    
                    axes[2].imshow(overlay)
                    axes[2].set_title("Overlay")
                    axes[2].axis('off')
                    
                    cam_short = cam_key.split('.')[-1]
                    plt.suptitle(f"Denoising Step {step_num} - {cam_short}")
                    plt.tight_layout()
                    
                    save_path = os.path.join(step_dir, f"{cam_short}_attention.png")
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Error visualizing step {step_num}, camera {cam_key}: {e}")
        
        print(f"Denoising attention saved to {denoising_dir}")
    
    # 在 visualize_denoising_attention 方法之后，visualize_cross_camera_attention 方法之前添加

    def visualize_word_attention(
        self,
        attention_dict: Dict[str, torch.Tensor],
        images: Dict[str, Image.Image],
        instruction: str
    ):
        """可视化每个词对图像的注意力 - 细粒度语言 grounding 分析"""
        if not attention_dict or self.sequence_layout is None:
            print("Cannot visualize word attention: missing data or layout")
            return
        
        word_dir = os.path.join(self.current_output_dir, "word_attention")
        os.makedirs(word_dir, exist_ok=True)
        
        # Tokenize 指令获取每个词
        tokens = self.processor.tokenizer(
            instruction,
            return_tensors="pt",
            add_special_tokens=True
        )
        token_ids = tokens["input_ids"][0].tolist()
        token_words = [self.processor.tokenizer.decode([tid]) for tid in token_ids]
        
        # 使用最后一层
        layer_names = sorted(attention_dict.keys(), key=lambda x: int(x.split('_')[-1]))
        attn = attention_dict[layer_names[-1]]
        
        if attn.dim() == 4:
            attn_avg = attn[0].mean(0).float().numpy()
        else:
            attn_avg = attn.mean(0).float().numpy()
        
        layout = self.sequence_layout
        text_start, text_end = layout.text_token_range
        
        # 选择关键词（非 padding，非 special tokens）
        key_words = []
        special_tokens = {'<pad>', '<s>', '</s>', '<unk>', '<|endoftext|>', '', ' '}
        
        for idx, (tid, word) in enumerate(zip(token_ids, token_words)):
            word_clean = word.strip()
            if word_clean and word_clean not in special_tokens:
                key_words.append((idx, word_clean))
        
        if not key_words:
            print("No valid words found for word attention visualization")
            return
        
        # 限制最多显示 10 个词
        if len(key_words) > 10:
            key_words = key_words[:10]
        
        print(f"Visualizing attention for words: {[w[1] for w in key_words]}")
        
        for cam_key, image in images.items():
            img_start, img_end = layout.image_token_ranges[cam_key]
            
            n_words = len(key_words)
            cols = min(5, n_words)
            rows = (n_words + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for word_idx, (token_idx, word) in enumerate(key_words):
                row, col = word_idx // cols, word_idx % cols
                ax = axes[row, col]
                
                # 该词对图像的注意力
                text_pos = text_start + token_idx
                if text_pos < text_end and text_pos < attn_avg.shape[0]:
                    word_to_img = attn_avg[text_pos, img_start:img_end]
                    
                    attn_2d = self._reshape_to_2d(word_to_img, layout.patches_per_side)
                    attn_2d = self._normalize_attention(attn_2d)
                    
                    overlay = self._create_overlay(image, attn_2d, 0.6)
                    ax.imshow(overlay)
                else:
                    ax.imshow(np.array(image))
                
                # 清理词显示（去除特殊字符）
                display_word = word.replace('▁', '').replace('Ġ', '').strip()
                if not display_word:
                    display_word = word
                ax.set_title(f'"{display_word}"', fontsize=11, fontweight='bold')
                ax.axis('off')
            
            # 隐藏空白子图
            for idx in range(len(key_words), rows * cols):
                row, col = idx // cols, idx % cols
                axes[row, col].axis('off')
            
            cam_short = cam_key.split('.')[-1]
            plt.suptitle(f"Per-Word Attention: Which region does each word attend to?\n({cam_short})", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(word_dir, f"{cam_short}_word_attention.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # 额外：生成词注意力强度对比图
        self._visualize_word_attention_comparison(attention_dict, images, key_words, instruction)
        
        print(f"Word attention saved to {word_dir}")
    
    def _visualize_word_attention_comparison(
        self,
        attention_dict: Dict[str, torch.Tensor],
        images: Dict[str, Image.Image],
        key_words: List[Tuple[int, str]],
        instruction: str
    ):
        """生成词注意力强度对比图"""
        if not key_words:
            return
        
        word_dir = os.path.join(self.current_output_dir, "word_attention")
        
        layer_names = sorted(attention_dict.keys(), key=lambda x: int(x.split('_')[-1]))
        attn = attention_dict[layer_names[-1]]
        
        if attn.dim() == 4:
            attn_avg = attn[0].mean(0).float().numpy()
        else:
            attn_avg = attn.mean(0).float().numpy()
        
        layout = self.sequence_layout
        text_start, text_end = layout.text_token_range
        
        camera_keys = list(images.keys())
        n_cams = len(camera_keys)
        n_words = len(key_words)
        
        # 计算每个词对每个相机的总注意力
        word_cam_attention = np.zeros((n_words, n_cams))
        
        for w_idx, (token_idx, word) in enumerate(key_words):
            text_pos = text_start + token_idx
            if text_pos >= text_end or text_pos >= attn_avg.shape[0]:
                continue
            
            for c_idx, cam_key in enumerate(camera_keys):
                img_start, img_end = layout.image_token_ranges[cam_key]
                word_to_cam = attn_avg[text_pos, img_start:img_end].sum()
                word_cam_attention[w_idx, c_idx] = word_to_cam
        
        # 归一化
        if word_cam_attention.max() > 0:
            word_cam_attention = word_cam_attention / word_cam_attention.max()
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(max(8, n_cams * 2), max(6, n_words * 0.5)))
        
        im = ax.imshow(word_cam_attention, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        word_labels = [w[1].replace('▁', '').replace('Ġ', '').strip() or w[1] for w in key_words]
        cam_labels = [k.split('.')[-1] for k in camera_keys]
        
        ax.set_xticks(range(n_cams))
        ax.set_yticks(range(n_words))
        ax.set_xticklabels(cam_labels, fontsize=10)
        ax.set_yticklabels(word_labels, fontsize=10)
        
        ax.set_xlabel("Camera", fontsize=12)
        ax.set_ylabel("Word", fontsize=12)
        ax.set_title("Word-to-Camera Attention Strength", fontsize=14, fontweight='bold')
        
        # 添加数值标注
        for i in range(n_words):
            for j in range(n_cams):
                val = word_cam_attention[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                       fontsize=9, color=color)
        
        plt.colorbar(im, ax=ax, label="Normalized Attention")
        plt.tight_layout()
        
        save_path = os.path.join(word_dir, "word_camera_heatmap.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def visualize_cross_camera_attention(
        self,
        attention_dict: Dict[str, torch.Tensor],
        images: Dict[str, Image.Image]
    ):
        """可视化相机之间的相互关注"""
        if not attention_dict or len(images) < 2:
            print("Skipping cross-camera visualization: not enough cameras or attention data")
            return
        
        cross_cam_dir = os.path.join(self.current_output_dir, "cross_camera")
        os.makedirs(cross_cam_dir, exist_ok=True)
        
        layer_names = sorted(attention_dict.keys(), key=lambda x: int(x.split('_')[-1]))
        if not layer_names:
            return
        
        last_layer = attention_dict[layer_names[-1]]
        layout = self.sequence_layout
        
        if layout is None:
            return
        
        if last_layer.dim() == 4:
            attn = last_layer[0]
        else:
            attn = last_layer
        
        attn_avg = attn.mean(0).float().numpy()
        seq_len = attn_avg.shape[0]
        
        camera_keys = list(images.keys())
        n_cams = len(camera_keys)
        cross_attn_matrix = np.zeros((n_cams, n_cams))
        
        for i, src_cam in enumerate(camera_keys):
            if src_cam not in layout.image_token_ranges:
                continue
            src_start, src_end = layout.image_token_ranges[src_cam]
            if src_start >= seq_len or src_end > seq_len:
                continue
            
            for j, tgt_cam in enumerate(camera_keys):
                if tgt_cam not in layout.image_token_ranges:
                    continue
                tgt_start, tgt_end = layout.image_token_ranges[tgt_cam]
                if tgt_start >= seq_len or tgt_end > seq_len:
                    continue
                
                cam_attn = attn_avg[src_start:src_end, tgt_start:tgt_end]
                if cam_attn.size > 0:
                    cross_attn_matrix[i, j] = cam_attn.mean()
        
        cross_attn_matrix = np.nan_to_num(cross_attn_matrix, nan=0.0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        vmin = cross_attn_matrix.min()
        vmax = cross_attn_matrix.max()
        if vmax == vmin:
            vmax = vmin + 1e-6
        
        im = ax.imshow(cross_attn_matrix, cmap='YlOrRd', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(range(n_cams))
        ax.set_yticks(range(n_cams))
        
        short_names = [k.split('.')[-1] for k in camera_keys]
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.set_yticklabels(short_names)
        
        ax.set_xlabel("Attended Camera (Key)")
        ax.set_ylabel("Attending Camera (Query)")
        ax.set_title("Cross-Camera Attention (VLM Last Layer)")
        
        for i in range(n_cams):
            for j in range(n_cams):
                val = cross_attn_matrix[i, j]
                ax.text(j, i, f"{val:.4f}", ha='center', va='center', fontsize=10,
                       color='white' if val > (vmax + vmin) / 2 else 'black')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        save_path = os.path.join(cross_cam_dir, "cross_camera_attention.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Cross-camera attention saved to {save_path}")
    
    def compute_attention_statistics(
        self,
        attention_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """计算注意力统计数据"""
        if not attention_dict or self.sequence_layout is None:
            return {}
        
        layout = self.sequence_layout
        stats = {}
        
        for layer_name, attn in attention_dict.items():
            if attn.dim() == 4:
                attn_np = attn[0].mean(0).float().numpy()
            else:
                attn_np = attn.mean(0).float().numpy()
            
            layer_stats = {}
            
            text_start, text_end = layout.text_token_range
            total_img_attn = 0
            for cam_key, (img_start, img_end) in layout.image_token_ranges.items():
                text_to_img = attn_np[text_start:text_end, img_start:img_end].sum()
                total_img_attn += text_to_img
                layer_stats[f"text_to_{cam_key.split('.')[-1]}"] = float(text_to_img)
            
            layer_stats["text_to_all_images"] = float(total_img_attn)
            
            state_start, state_end = layout.state_token_range
            state_attn = attn_np[:, state_start:state_end].sum()
            layer_stats["total_to_state"] = float(state_attn)
            
            for cam_key, (img_start, img_end) in layout.image_token_ranges.items():
                self_attn = attn_np[img_start:img_end, img_start:img_end].mean()
                layer_stats[f"{cam_key.split('.')[-1]}_self_attention"] = float(self_attn)
            
            stats[layer_name] = layer_stats
        
        import json
        stats_path = os.path.join(self.current_output_dir, "attention_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {stats_path}")
        return stats
    
    # 找到 visualize_all 方法并修改

    def visualize_all(
        self,
        attention_data: Dict[str, Dict[str, torch.Tensor]],
        images: Dict[str, Image.Image],
        instruction: str
    ):
        """完整可视化"""
        print(f"\nSaving visualizations to: {self.current_output_dir}")
        
        info_path = os.path.join(self.current_output_dir, "input_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Instruction: {instruction}\n")
            f.write(f"Cameras: {list(images.keys())}\n")
            f.write(f"Image sizes: {[img.size for img in images.values()]}\n")
        
        vlm_attention = attention_data.get("vlm", {})
        denoising_attention = attention_data.get("denoising", {})
        
        print(f"VLM layers: {len(vlm_attention)}")
        if denoising_attention:
            num_steps = len(set(k.split('_')[1] for k in denoising_attention.keys()))
            print(f"Denoising steps: {num_steps}")
        
        # 1. VLM 注意力（Text→Image, Cross-Camera）
        if vlm_attention:
            self.visualize_vlm_attention(vlm_attention, images)
        
        # 2. Denoising 注意力（Action→Image）
        if denoising_attention:
            self.visualize_denoising_attention(denoising_attention, images)
        
        # 3. 逐词注意力分析（细粒度 Language Grounding）
        if vlm_attention:
            self.visualize_word_attention(vlm_attention, images, instruction)
        
        # 4. 跨相机注意力矩阵
        if vlm_attention and len(images) > 1:
            self.visualize_cross_camera_attention(vlm_attention, images)
        
        # 5. 统计数据
        if vlm_attention:
            self.compute_attention_statistics(vlm_attention)
        
        print(f"\nVisualization complete!")
    
    def cleanup(self):
        """清理资源"""
        if self.policy is not None:
            del self.policy
        torch.cuda.empty_cache()


def main():
    """测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output", type=str, default="./myscripts/attention/outputs")
    
    args = parser.parse_args()
    
    config = AttentionConfig(
        model_path=args.model,
        output_dir=args.output
    )
    
    visualizer = SmolVLAAttentionVisualizer(config)
    
    try:
        visualizer.load_model()
        
        if args.repo_id:
            from run_visualization import load_dataset_item
            
            data = load_dataset_item(
                repo_id=args.repo_id,
                episode_index=args.episode_index,
                frame_index=args.frame_index
            )
            
            images = data['images']
            state = data['state']
            instruction = data['instruction']
            stats = data['stats']
            
            visualizer.set_normalization_stats(stats)
            
            image_name = f"ep{args.episode_index:03d}_frame{args.frame_index:04d}"
            
        else:
            images = {"camera": Image.new('RGB', (640, 480), (100, 150, 200))}
            state = None
            instruction = "Pick up the red cube."
            image_name = "test"
        
        attention_data, actions = visualizer.extract_attention_with_output(
            images=images,
            instruction=instruction,
            state=state,
            image_name=image_name
        )
        
        visualizer.visualize_all(attention_data, images, instruction)
        
        print(f"\nPredicted actions shape: {actions.shape}")
        
    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    main()