from dataclasses import asdict, dataclass, field
import abc
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import itertools
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.embeddings import apply_rotary_emb
from enum import IntEnum


@dataclass
class FeatureAlignConfig:
    """Config for feature analogical alignment in attention value routing."""

    start_inject: int = 0
    end_inject: int = 5
    lambda_shift: float = 2.0
    use_soft_mask: bool = False


class TransType(IntEnum):
    DOUBLE = 0
    SINGLE = 1


# copied from diffusers.models.transformers.transformer_flux._get_projections
def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


# copied from diffusers.models.transformers.transformer_flux._get_fused_projections
def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


# copied from diffusers.models.transformers.transformer_flux._get_qkv_projections
def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class FeatureAlignFluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, pipe, prompt, out_width, out_height, config: Optional[FeatureAlignConfig] = None):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.block_idx = [0, 0]
        self.step_idx = 0
        self.batch_size = len(prompt)
        self.out_width = out_width
        self.out_height = out_height
        self.config = config if config is not None else FeatureAlignConfig()

        print(f"Injecting [feature align] Attention Processor...")
        for i, tblock in enumerate(pipe.transformer.transformer_blocks):
            tblock.attn.set_processor(self)
        for i, tblock in enumerate(pipe.transformer.single_transformer_blocks):
            tblock.attn.set_processor(self)
        self.attention_store = None
        self.seq_len = None
        self.text_seq_len = None
        self.latent_seq_len = None

        self.attn_save_counter = 0

    def _compute_grid_shape(self) -> Tuple[int, int, int, int]:
        if self.latent_seq_len is None:
            raise ValueError("latent_seq_len is not initialized yet")
        if self.out_width <= 0 or self.out_height <= 0:
            raise ValueError(f"Invalid output size: {self.out_width}x{self.out_height}")

        ratio = self.out_height / self.out_width
        target_h = math.sqrt(self.latent_seq_len * ratio)

        best = None
        best_score = float("inf")
        max_divisor = int(math.sqrt(self.latent_seq_len))
        for h in range(1, max_divisor + 1):
            if self.latent_seq_len % h != 0:
                continue
            w = self.latent_seq_len // h
            for cand_h, cand_w in ((h, w), (w, h)):
                if cand_h % 2 != 0 or cand_w % 2 != 0:
                    continue
                ratio_err = abs((cand_h / cand_w) - ratio)
                target_err = abs(cand_h - target_h) / max(target_h, 1.0)
                score = ratio_err + 0.1 * target_err
                if score < best_score:
                    best_score = score
                    best = (cand_h, cand_w)

        if best is None:
            raise ValueError(
                f"Cannot split latent seq_len={self.latent_seq_len} into even HxW for 2x2 grid"
            )

        H_p, W_p = best
        return H_p, W_p, H_p // 2, W_p // 2

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block_type = TransType.SINGLE if encoder_hidden_states is None else TransType.DOUBLE
        if block_type == TransType.DOUBLE:
            num_blocks = len(self.pipe.transformer.transformer_blocks)
        elif block_type == TransType.SINGLE:
            num_blocks = len(self.pipe.transformer.single_transformer_blocks)
        if self.text_seq_len is None and block_type == TransType.DOUBLE:
            self.text_seq_len = encoder_hidden_states.shape[1]
            self.latent_seq_len = hidden_states.shape[1]
            self.seq_len = self.text_seq_len + self.latent_seq_len

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # =====================================================================
        # 🌟 2x2 Grid 无 RoPE 语义路由与动态对齐代数
        # =====================================================================
        
        start_inject = self.config.start_inject
        end_inject = self.config.end_inject
        lambda_shift = self.config.lambda_shift
        
        if block_type == TransType.DOUBLE and start_inject <= self.step_idx < end_inject:
            img_q = query[:, self.text_seq_len:, :, :].clone()
            img_k = key[:, self.text_seq_len:, :, :].clone()
            img_v = value[:, self.text_seq_len:, :, :].clone()

            # 🌟 1. 动态自适应长宽比计算
            ratio = self.out_height / self.out_width
            H_p = int(math.sqrt(self.latent_seq_len * ratio))
            W_p = self.latent_seq_len // H_p
            
            # 单张子图的高和宽
            h_p = H_p // 2
            w_p = W_p // 2

            q_grid = img_q.view(1, H_p, W_p, attn.heads, attn.head_dim)
            k_grid = img_k.view(1, H_p, W_p, attn.heads, attn.head_dim)
            v_grid = img_v.view(1, H_p, W_p, attn.heads, attn.head_dim)

            # 🌟 2. 使用动态高宽 (h_p, w_p) 进行精准切片
            # A (左上): h_p x w_p
            k_A = k_grid[0, :h_p, :w_p, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)
            v_A = v_grid[0, :h_p, :w_p, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)

            # A' (右上): h_p x w_p
            v_A_prime = v_grid[0, :h_p, w_p:, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)

            # B (左下): h_p x w_p
            q_B = q_grid[0, h_p:, :w_p, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)
            v_B = v_grid[0, h_p:, :w_p, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)

            # 🧠 无 RoPE 语义寻路
            scale = attn.head_dim ** -0.5
            sim = torch.bmm(q_B, k_A.transpose(1, 2)) * scale
            attn_weights = torch.softmax(sim, dim=-1)

            aligned_v_A = torch.bmm(attn_weights, v_A)            
            aligned_v_A_prime = torch.bmm(attn_weights, v_A_prime) 

            delta_v_aligned = aligned_v_A_prime - aligned_v_A
            
            # (预留：你可以根据 self.config.align.use_soft_mask 决定是否开启软掩膜)
            
            v_syn_h = v_B + lambda_shift * delta_v_aligned
            
            # 🌟 3. 覆写位置适配 (h_p, w_p)
            v_syn_grid = v_syn_h.permute(1, 0, 2).view(h_p, w_p, attn.heads, attn.head_dim)
            
            # 右下 B'
            v_grid[0, h_p:, w_p:, :, :] = v_syn_grid
            
            # 左下 B (根据你之前的实验，如果要覆写)
            v_grid[0, h_p:, :w_p, :, :] = v_syn_grid

            v_img_new = v_grid.view(1, self.latent_seq_len, attn.heads, attn.head_dim)
            value = torch.cat([value[:, :self.text_seq_len, :, :], v_img_new], dim=1)
        # =====================================================================
        # 恢复代码，继续执行 RoPE 和正常的 Self-Attention
        # =====================================================================
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            
        # ... 后续原封不动的 Attention 计算 ...

        # q k v shape: (batch_size, seq_len, heads, head_dim) [1, 4608, 24, 128]
        # to use attn.get_attention_scores() we need to reshape to (batch_size * heads, seq_len, head_dim)
        batch_size, seq_len, heads, head_dim = query.shape
        assert seq_len == self.seq_len
        query = query.permute(0, 2, 1, 3).reshape(-1, query.shape[1], query.shape[3])
        key = key.permute(0, 2, 1, 3).reshape(-1, key.shape[1], key.shape[3])
        value = value.permute(0, 2, 1, 3).reshape(-1, value.shape[1], value.shape[3])

        # the original dispatch_attention_fn uses torch.nn.functional.scaled_dot_product_attention
        # which is highly optimized to be numerically stable in float16 or bfloat16 by using online softmax(?)
        # we don't have that. So we upcast to float32 here to be safe.
        attn.upcast_attention = True
        attn.upcast_softmax = True

        attn.scale = attn.head_dim**-0.5

        attention_probs = attn.get_attention_scores(
            query,
            key,
            attention_mask=attention_mask,
        )

        # =====================================================================
        # 🌟 Store Image-to-Image Attention Maps
        # =====================================================================
        if block_type == TransType.DOUBLE:
            
            # 【核心修改点】：在这里直接将其转换为 float32
            attention_probs_fp32 = attention_probs.to(torch.float32)
            
            # Reshape to separate batch and heads: (batch_size, heads, S_img, S_img)
            attn_to_save = attention_probs_fp32.view(batch_size, -1, self.seq_len, self.seq_len)

            # Average across all heads to reduce memory: (batch_size, S_img, S_img)
            attn_avg = attn_to_save.mean(dim=1)
            
            # assert softmax along the last dimension sums to 1
            assert torch.allclose(attn_to_save.sum(dim=-1), torch.ones_like(attn_to_save.sum(dim=-1)), rtol=0.01)

            # Accumulate the attention maps
            if self.attention_store is None:
                self.attention_store = attn_avg
            else:
                self.attention_store += attn_avg
            self.attn_save_counter += 1
        # =====================================================================

        hidden_states = torch.bmm(attention_probs, value)
        # hidden_states shape: (batch_size * heads, seq_len, head_dim)
        # reshape back to (batch_size, seq_len, heads, head_dim)
        hidden_states = hidden_states.view(batch_size, attn.heads, seq_len, attn.head_dim).permute(0, 2, 1, 3)
        # flatten heads and head_dim
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        ret = None
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            ret = hidden_states, encoder_hidden_states
        else:
            ret = hidden_states

        self.block_idx[block_type] += 1
        if block_type is TransType.SINGLE and self.block_idx[block_type] >= num_blocks:
            self.block_idx = [0, 0]
            self.step_idx += 1
        return ret