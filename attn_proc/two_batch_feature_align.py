import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import itertools
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.embeddings import apply_rotary_emb
from enum import IntEnum


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

    def __init__(self, pipe, prompt, out_width, out_height):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.block_idx = [0, 0]
        self.step_idx = 0
        self.batch_size = len(prompt)
        self.out_width = out_width
        self.out_height = out_height

        print("Injecting [2 batch feature align] Attention Processor into the pipeline...")
        for i, tblock in enumerate(pipe.transformer.transformer_blocks):
            tblock.attn.set_processor(self)
        for i, tblock in enumerate(pipe.transformer.single_transformer_blocks):
            tblock.attn.set_processor(self)
        self.attention_store = None
        self.seq_len = None
        self.text_seq_len = None
        self.latent_seq_len = None

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
        # 🌟 终极魔改区：无 RoPE 语义路由与动态对齐代数
        # 必须在 apply_rotary_emb 之前执行，利用纯语义特征进行空间对齐！
        # =====================================================================
        
        start_inject = 15   # 前 5 步让初始噪声稳定
        end_inject = 35    # 后 15 步交还给模型自己画油画质感
        
        if start_inject <= self.step_idx < end_inject:
            import math
            # 1. 剥离文本 Token，只拿图像 Token
            img_q = query[:, self.text_seq_len:, :, :].clone()
            img_k = key[:, self.text_seq_len:, :, :].clone()
            img_v = value[:, self.text_seq_len:, :, :].clone()

            # 因为是两张图水平拼接 [A | A']，所以宽是高的两倍: W_p = 2 * H_p
            # S_img = H_p * (2 * H_p) = 2 * H_p^2  => H_p = sqrt(S_img / 2)
            H_p = int(math.sqrt(self.latent_seq_len / 2))
            W_p = H_p * 2
            half_w = W_p // 2

            # 重塑为 (Batch=2, H_p, W_p, heads, head_dim)
            # 这就把 1D 的拉链序列，完美还原成了 2D 物理空间图像！
            q_grid = img_q.view(2, H_p, W_p, attn.heads, attn.head_dim)
            k_grid = img_k.view(2, H_p, W_p, attn.heads, attn.head_dim)
            v_grid = img_v.view(2, H_p, W_p, attn.heads, attn.head_dim)

            # 3. 在 2D 空间上精准切片 (只切 W 维度)，再变回 BMM 需要的 (heads, seq, dim)
            # Batch 0: A (左) 和 A' (右)
            k_A = k_grid[0, :, :half_w, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)
            v_A = v_grid[0, :, :half_w, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)
            v_A_prime = v_grid[0, :, half_w:, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)

            # Batch 1: B (左) 和 B' (右，B'目前还是原声背景，不提取)
            q_B = q_grid[1, :, :half_w, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)
            v_B = v_grid[1, :, :half_w, :, :].reshape(-1, attn.heads, attn.head_dim).permute(1, 0, 2)

            # 4. 🧠 无 RoPE 语义寻路 (找对应像素点)
            scale = attn.head_dim ** -0.5
            sim = torch.bmm(q_B, k_A.transpose(1, 2)) * scale
            attn_weights = torch.softmax(sim, dim=-1)

            # 5. 特征重组与对齐任务代数
            aligned_v_A = torch.bmm(attn_weights, v_A)            
            aligned_v_A_prime = torch.bmm(attn_weights, v_A_prime) 

            lambda_shift = 3.1  
            delta_v_aligned = aligned_v_A_prime - aligned_v_A
            
            # 完美贴合油画苹果骨架的油画橘子特征
            v_syn_h = v_B + lambda_shift * delta_v_aligned
            
            # 6. 覆写到 Value 2D 矩阵的 B' (Batch 1 右半边) 位置
            v_syn_grid = v_syn_h.permute(1, 0, 2).view(H_p, half_w, attn.heads, attn.head_dim)
            v_grid[1, :, half_w:, :, :] = v_syn_grid

            # 7. 把 2D 网格重新压扁回 1D，拼回原来的大 Tensor 中
            v_img_new = v_grid.view(2, self.latent_seq_len, attn.heads, attn.head_dim)
            value = torch.cat([value[:, :self.text_seq_len, :, :], v_img_new], dim=1)

            # (可选) 也可以把 K 同步覆写，但通常修改 V 已经足够主导生成内容
            # k_syn_h = k_B + lambda_shift * (torch.bmm(attn_weights, k_A_prime.permute(2,0,1)) - torch.bmm(attn_weights, k_A.permute(2,0,1)))
            # key[1, self.text_seq_len + mid:, :, :] = k_syn_h.permute(1, 2, 0)

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
        # Extract L2L part, shape: (batch_size * heads, S_img, S_img)
        l2l_attn = attention_probs[:, self.text_seq_len:, self.text_seq_len:]

        # Reshape to separate batch and heads: (batch_size, heads, S_img, S_img)
        l2l_attn = l2l_attn.view(self.batch_size, -1, self.latent_seq_len, self.latent_seq_len)
        
        # Average across all heads to reduce memory: (batch_size, S_img, S_img)
        l2l_attn_avg = l2l_attn.mean(dim=1)
        
        # Accumulate the attention maps
        if self.attention_store is None:
            self.attention_store = l2l_attn_avg
        else:
            self.attention_store += l2l_attn_avg
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