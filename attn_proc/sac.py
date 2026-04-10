import abc
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
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


class SACFluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, pipe, prompt, out_width, out_height, end_step=30, b_a_copyto_bp_ap_tau: float = 1.6, b_b_copyto_bp_b_tau: float = 1.0):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.block_idx = [0, 0]
        self.step_idx = 0
        self.batch_size = len(prompt) if isinstance(prompt, list) else 1
        self.out_width = out_width
        self.out_height = out_height

        self.end_step = end_step
        self.b_a_copyto_bp_ap_tau = b_a_copyto_bp_ap_tau
        self.b_b_copyto_bp_b_tau = b_b_copyto_bp_b_tau

        print("Injecting [SAC] Attention Processor into the pipeline...")
        for i, tblock in enumerate(pipe.transformer.transformer_blocks):
            tblock.attn.set_processor(self)
        for i, tblock in enumerate(pipe.transformer.single_transformer_blocks):
            tblock.attn.set_processor(self)
        self.attention_store = None
        self.seq_len = None
        self.text_seq_len = None
        self.latent_seq_len = None

        self.attn_save_counter = 0

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
        # 拦截区：在正常 RoPE 之前，保存未加位置编码的 Query (用于你的伪造方案)
        # 注意此时 query_unrope 是 4D: (batch_size, seq_len, heads, head_dim)
        # =====================================================================
        query_unrope = query.clone()

        # 1. 正常应用 RoPE (给主体生成用)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # q k v shape: (batch_size, seq_len, heads, head_dim)
        batch_size, seq_len, heads, head_dim = query.shape
        assert seq_len == self.seq_len
        
        # 将 q k v 拍平为 3D: (batch_size * heads, seq_len, head_dim) 以计算注意力
        query = query.permute(0, 2, 1, 3).reshape(-1, query.shape[1], query.shape[3])
        key = key.permute(0, 2, 1, 3).reshape(-1, key.shape[1], key.shape[3])
        value = value.permute(0, 2, 1, 3).reshape(-1, value.shape[1], value.shape[3])

        # 2. 计算原生的 pre-softmax 注意力分数
        scale = attn.head_dim**-0.5
        sim = torch.bmm(query, key.transpose(-1, -2)) * scale

        # =====================================================================
        # 🌟 核心魔改区：Self-Attention Cloning (SAC) 矩阵手术
        # =====================================================================
        inject_threshold = self.end_step  

        if self.step_idx < inject_threshold:
            import math
            latent_seq_len = self.latent_seq_len

            ratio = self.out_height / self.out_width
            H_p = int(math.sqrt(self.latent_seq_len * ratio))
            W_p = self.latent_seq_len // H_p
            assert abs(ratio - H_p / W_p) < 0.01, "Calculated grid size does not match the specified aspect ratio"
            assert H_p * W_p == latent_seq_len, "Calculated grid size does not match latent sequence length"

            # 单张子图的高和宽
            h_p = H_p // 2
            w_p = W_p // 2

            # 将图像部分的注意力矩阵重塑为 5D: (batch*heads, Q_y, Q_x, K_y, K_x)
            img_sim = sim[:, self.text_seq_len:, self.text_seq_len:].view(-1, H_p, W_p, H_p, W_p)

            # --- 按照你的思路：构造带有 B' 位置的 Q_B ---

            # 【A. 获取 B' 的 RoPE】(兼容处理)
            if isinstance(image_rotary_emb, tuple):
                cos_emb, sin_emb = image_rotary_emb
                if cos_emb.dim() == 2:
                    img_rope_cos = cos_emb[self.text_seq_len:, :].view(H_p, W_p, cos_emb.shape[-1])
                    img_rope_sin = sin_emb[self.text_seq_len:, :].view(H_p, W_p, sin_emb.shape[-1])
                    rope_b_prime_cos = img_rope_cos[h_p:, w_p:, :].reshape(h_p * w_p, cos_emb.shape[-1])
                    rope_b_prime_sin = img_rope_sin[h_p:, w_p:, :].reshape(h_p * w_p, sin_emb.shape[-1])
                else:
                    img_rope_cos = cos_emb[:, self.text_seq_len:, :].view(-1, H_p, W_p, cos_emb.shape[-1])
                    img_rope_sin = sin_emb[:, self.text_seq_len:, :].view(-1, H_p, W_p, sin_emb.shape[-1])
                    rope_b_prime_cos = img_rope_cos[:, h_p:, w_p:, :].reshape(-1, h_p * w_p, cos_emb.shape[-1])
                    rope_b_prime_sin = img_rope_sin[:, h_p:, w_p:, :].reshape(-1, h_p * w_p, sin_emb.shape[-1])
                rope_b_prime = (rope_b_prime_cos, rope_b_prime_sin)
            else:
                if image_rotary_emb.dim() == 2:
                    img_rope = image_rotary_emb[self.text_seq_len:, :].view(H_p, W_p, image_rotary_emb.shape[-1])
                    rope_b_prime = img_rope[h_p:, w_p:, :].reshape(h_p * w_p, image_rotary_emb.shape[-1])
                else:
                    img_rope = image_rotary_emb[:, self.text_seq_len:, :].view(-1, H_p, W_p, image_rotary_emb.shape[-1])
                    rope_b_prime = img_rope[:, h_p:, w_p:, :].reshape(-1, h_p * w_p, image_rotary_emb.shape[-1])

            # 【B. 提取未加 RoPE 的 B 区域 Query (Q_B)】
            # query_unrope 是 4D 的: (batch, seq_len, heads, head_dim)
            img_q_unrope = query_unrope[:, self.text_seq_len:, :, :].view(-1, H_p, W_p, heads, head_dim)
            q_unrope_b = img_q_unrope[:, h_p:, :w_p, :, :].reshape(-1, h_p * w_p, heads, head_dim)

            # 【C. RoPE Deception：给 Q_B 穿上 B' 的位置外衣】
            # 输出的 q_fake_b_prime 是 4D: (batch, h_p*w_p, heads, head_dim)
            q_fake_b_prime = apply_rotary_emb(q_unrope_b, rope_b_prime, sequence_dim=1)

            # 将伪造的 Query 压平为 3D: (batch*heads, h_p*w_p, head_dim)
            q_fake_b_prime = q_fake_b_prime.permute(0, 2, 1, 3).reshape(-1, h_p * w_p, head_dim)

            # 【D. 提取正常的 B 区域 Key (K_B) 】
            # 此时的 key 已经是 3D (batch*heads, seq, dim) 且带有原始的 RoPE_B
            img_k = key[:, self.text_seq_len:, :].view(-1, H_p, W_p, head_dim)
            k_b = img_k[:, h_p:, :w_p, :].reshape(-1, h_p * w_p, head_dim)

            # 【E. 计算物理规律正确的完美注意力分数】
            sim_fake = torch.bmm(q_fake_b_prime, k_b.transpose(-1, -2)) * scale
            sim_fake = sim_fake.view(-1, h_p, w_p, h_p, w_p)

            sim_fake = sim_fake + self.b_b_copyto_bp_b_tau
            
            # 🌟 直接覆盖 B' -> B 的注意力！不需要加 tau 补偿了！
            img_sim[:, h_p:, w_p:, h_p:, :w_p] = sim_fake
            
            # ... 继续执行你的 B'->A' 或者其他的克隆逻辑 ...

            # --- 矩阵克隆开始 ---
            # 目标: B' (右下, y:h_p~end, x:w_p~end)

            # 动作 1: 语义与结构克隆 (SAC 真正的灵魂)
            # 将 B' -> A' 的注意力分数直接改成与 B -> A 相同
            # 这样 B' 在尝试画橘子时，会完美套用原版苹果的结构骨架！
            clone_b_to_a = img_sim[:, h_p:, :w_p, :h_p, :w_p].clone()
            native_b_prime_to_a_prime = img_sim[:, h_p:, w_p:, :h_p, w_p:].clone()
            max_native_b_prime_to_a_prime = native_b_prime_to_a_prime.amax(dim=-1, keepdim=True)
            max_b_to_a = clone_b_to_a.amax(dim=-1, keepdim=True)
            # 🌟 动态对齐 (Logit Alignment)
            aligned_b_prime_to_a_prime = clone_b_to_a - max_b_to_a + max_native_b_prime_to_a_prime + self.b_a_copyto_bp_ap_tau
            img_sim[:, h_p:, w_p:, :h_p, w_p:] = aligned_b_prime_to_a_prime

            # 动作 2: 阻断旧语义源 (防止画出红苹果)
            # 严禁 B' 看 A (左上的原苹果)，逼迫它只能顺着动作1去 A' 吸取橘子特征
            # img_sim[:, h_p:, w_p:, :h_p, :w_p] = -10000.0

            # ⚠️ 极其关键的修复：什么都不做！
            # 绝对不要动 B' -> B'：让它自己平滑噪声，生成完美的油画笔触！
            # 绝对不要动 B' -> B：让它利用原生 RoPE，自然地与左边的桌布和背景融合！
            # --- 矩阵克隆结束 ---

            # 将手术后的注意力矩阵展平放回原处
            sim[:, self.text_seq_len:, self.text_seq_len:] = img_sim.view(-1, latent_seq_len, latent_seq_len)

        # =====================================================================
        # 继续正常的 Softmax 和 Value 聚合
        # =====================================================================
        attn.upcast_attention = True
        attn.upcast_softmax = True

        # Softmax 会自动处理我们塞进去的 -10000.0，使其概率归零
        attention_probs = sim.softmax(dim=-1)

        # =====================================================================
        # 🌟 Store Image-to-Image Attention Maps
        # =====================================================================
        if block_type == TransType.DOUBLE:
            
            # 【核心修改点】：在这里直接将其转换为 float32
            attention_probs_fp32 = attention_probs.to(torch.float32)
            
            # Reshape to separate batch and heads: (batch_size, heads, S_img, S_img)
            attn_to_save = attention_probs_fp32.view(self.batch_size, -1, self.seq_len, self.seq_len)

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
        
        # ... 后面的维度还原代码保持原样 ...
        hidden_states = hidden_states.view(batch_size, attn.heads, seq_len, attn.head_dim).permute(0, 2, 1, 3)
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