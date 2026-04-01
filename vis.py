# %%
%matplotlib widget
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image

class InteractiveAttentionViewer:
    def __init__(
        self, 
        pt_path, 
        batch_idx=0,
        pixels_per_token=16
    ):
        self.pixels_per_token = pixels_per_token
        
        print("Loading data from checkpoint...")
        data_dict = torch.load(pt_path, weights_only=True)
        
        # Load dimensions and lengths
        self.target_width = data_dict["out_width"]
        self.target_height = data_dict["out_height"]
        self.text_seq_len = data_dict["text_seq_len"]
        self.latent_seq_len = data_dict["latent_seq_len"]
        self.tokens = data_dict["tokens"]
        
        self.h_tok = self.target_height // pixels_per_token
        self.w_tok = self.target_width // pixels_per_token
        self.total_tokens = self.h_tok * self.w_tok

        # Load Attention Matrix (seq_len x seq_len)
        print("Processing attention tensor...")
        attn_matrix = data_dict["attention_map"][batch_idx]
        if torch.is_tensor(attn_matrix):
            self.attn_matrix = attn_matrix.detach().cpu().float().numpy()
        else:
            self.attn_matrix = attn_matrix

        # Load Image
        print("Restoring image from tensor...")
        img_tensor = data_dict["image"][batch_idx]
        self.img_array = img_tensor.permute(1, 2, 0).numpy()

        print("Data loaded successfully! Initializing UI...")
        self.plot_top_k_attention_sinks(top_percent=3.5)
        self.setup_ui()

    def plot_top_k_attention_sinks(self, top_percent=5):
        print(f"Finding top {top_percent}% attention sinks based on Mean...")
        
        # 1. 提取 L2L
        l2l_matrix = self.attn_matrix[self.text_seq_len:, self.text_seq_len:]

        # 2. 计算基础指标
        mean_attn = l2l_matrix.mean(axis=0)
        min_attn = l2l_matrix.min(axis=0)
        
        # 2. 计算组合指标 (相乘)
        combined_attn = mean_attn * min_attn
        
        # 3. 计算百分位数阈值 (比如输入 5%，就是取第 95 百分位的值作为门槛)
        threshold = np.percentile(combined_attn, 100 - top_percent)
        
        # 4. 生成布尔掩码 (大于等于阈值的设为 True/1，否则为 False/0)
        sink_mask = combined_attn >= threshold
        
        # 5. 变形回 2D 图像网格
        combined_map_2d = combined_attn.reshape((self.h_tok, self.w_tok))
        mask_map_2d = sink_mask.reshape((self.h_tok, self.w_tok))
        
        # 6. 并排画图验证
        fig_top, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig_top.canvas.manager.set_window_title(f"Top {top_percent}% Attention Sinks")
        
        # --- Left: Original Mean ---
        im_mean = axes[0].imshow(np.log(combined_map_2d + 1e-6), cmap='inferno')
        axes[0].set_title("Mean x Min (Log)")
        axes[0].axis('off')
        fig_top.colorbar(im_mean, ax=axes[0], fraction=0.046, pad=0.04)
        
        # --- Right: Top X% Mask ---
        # 这是一个二值图，黄色的点就是我们找出来的前 X% 的 Sink tokens
        im_mask = axes[1].imshow(mask_map_2d, cmap='viridis') 
        axes[1].set_title(f"Top {top_percent}% Sinks Mask (Threshold: {threshold:.4f})")
        axes[1].axis('off')
        fig_top.colorbar(im_mask, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        # --- 新增：提取并打印黑洞的全局索引 ---
        # 1. 找到 mask 中为 True 的一维索引 (相对于图像 latent_seq_len)
        latent_sink_indices = np.where(sink_mask.flatten())[0]
        # 2. 加上 text_seq_len 偏移量，变成在整个 seq_len 中的全局索引
        global_sink_indices = latent_sink_indices + self.text_seq_len
        
        print(f"\n🚀 --- Attention Sinks Discovered ({top_percent}%) ---")
        print(f"Total sinks found: {len(global_sink_indices)}")
        print(f"Copy this list to your run.py:")
        # 打印成可以直接复制的 Python 列表格式
        print(f"SINK_INDICES = {global_sink_indices.tolist()}")

        plt.show()

    def plot_advanced_sink_metrics(self):
        print("Calculating sink metrics: Mean, Min, and Mean × Min...")
        
        # 1. 提取 L2L
        l2l_matrix = self.attn_matrix[self.text_seq_len:, self.text_seq_len:]
        
        # 2. 计算基础指标
        mean_attn = l2l_matrix.mean(axis=0)
        min_attn = l2l_matrix.min(axis=0)
        
        # 3. 计算组合指标 (相乘)
        combined_attn = mean_attn * min_attn
        
        # --- 变形回 2D ---
        mean_map_2d = mean_attn.reshape((self.h_tok, self.w_tok))
        min_map_2d = min_attn.reshape((self.h_tok, self.w_tok))
        combined_map_2d = combined_attn.reshape((self.h_tok, self.w_tok))
        
        # --- 画图 ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.canvas.manager.set_window_title("Attention Sink Detectors: Mean, Min, Product")
        
        # 1. Mean (Log)
        im0 = axes[0].imshow(np.log(mean_map_2d + 1e-8), cmap='inferno')
        axes[0].set_title("1. Mean Received Attention (Log)")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 2. Min (Log)
        im1 = axes[1].imshow(np.log(min_map_2d + 1e-8), cmap='inferno')
        axes[1].set_title("2. Minimum Received Attention (Log)")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Mean * Min (Log)
        # 注意这里用 1e-16 因为相乘后数值会更小
        im2 = axes[2].imshow(np.log(combined_map_2d + 1e-16), cmap='inferno')
        axes[2].set_title("3. Mean × Min (Log)")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

    def setup_ui(self):
        # 使用 GridSpec 创建 2x2 布局，下面一行合并用来显示柱状图
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("FLUX.1 Joint Attention Viewer")
        gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1])

        # --- Top Left: Original Image ---
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_img.imshow(self.img_array)
        self.ax_img.set_title("Image (Click patch to see L2L & L2T)")
        self.ax_img.axis('off')

        self.ax_img.axhline(self.target_height / 2, color='white', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_img.axvline(self.target_width / 2, color='white', linestyle='--', alpha=0.5, linewidth=1)

        self.rect = patches.Rectangle(
            (0, 0), self.pixels_per_token, self.pixels_per_token,
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
        )
        self.ax_img.add_patch(self.rect)

        # --- Top Right: Attention Map ---
        self.ax_attn = self.fig.add_subplot(gs[0, 1])
        init_attn_map = np.zeros((self.h_tok, self.w_tok))
        self.im_attn = self.ax_attn.imshow(init_attn_map, cmap='inferno')
        self.ax_attn.set_title("Attention Map (Waiting for click...)")
        self.ax_attn.axis('off')
        
        self.cbar = self.fig.colorbar(
            self.im_attn, ax=self.ax_attn, 
            fraction=0.046 * (self.target_height / self.target_width), pad=0.04
        )

        # --- Bottom: Text Tokens Bar Chart ---
        self.ax_bar = self.fig.add_subplot(gs[1, :])
        
        # 初始化 512 个柱子
        self.bars = self.ax_bar.bar(range(self.text_seq_len), np.zeros(self.text_seq_len), color='cornflowerblue')
        self.ax_bar.set_xlim(-1, self.text_seq_len)
        self.ax_bar.set_title("Text Token Activation (Click bar to see T2L on image)")

        # 优化 X 轴标签：过滤掉大量的 <pad>，只显示有意义的词
        valid_indices = [i for i, t in enumerate(self.tokens) if t not in ['<pad>', '</s>']]
        valid_tokens = [self.tokens[i].replace(' ', ' ') for i in valid_indices] # 清理 T5 潜在的特殊空格
        
        self.ax_bar.set_xticks(valid_indices)
        self.ax_bar.set_xticklabels(valid_tokens, rotation=45, ha='right', fontsize=9)
        
        # 文本柱状图的红色高亮框 (初始隐藏)
        self.bar_highlight = patches.Rectangle(
            (0, 0), 1, 0, linewidth=2, edgecolor='red', facecolor='none', alpha=0
        )
        self.ax_bar.add_patch(self.bar_highlight)

        # Bind events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.tight_layout()
        plt.show()

    def on_click(self, event):
        # ---------------------------------------------------------
        # 场景 A: 点击了上方的【图像区域】
        # ---------------------------------------------------------
        if event.inaxes == self.ax_img:
            x_pixel, y_pixel = event.xdata, event.ydata
            x_tok = int(x_pixel // self.pixels_per_token)
            y_tok = int(y_pixel // self.pixels_per_token)
            x_tok = max(0, min(x_tok, self.w_tok - 1))
            y_tok = max(0, min(y_tok, self.h_tok - 1))

            # 更新图片红框
            self.rect.set_xy((x_tok * self.pixels_per_token, y_tok * self.pixels_per_token))

            # 图像 Token 在整个 seq_len 中的全局索引 (需要加上 text_seq_len 偏移)
            img_token_idx = y_tok * self.w_tok + x_tok
            global_token_idx = self.text_seq_len + img_token_idx

            # 1. 提取 L2L (Image-to-Image) 并更新右上角热力图
            l2l_attn = self.attn_matrix[global_token_idx, self.text_seq_len:]
            l2l_map_2d = l2l_attn.reshape((self.h_tok, self.w_tok))
            log_l2l_map = np.log(l2l_map_2d + 1e-6)
            
            self.im_attn.set_data(log_l2l_map)
            self.im_attn.set_clim(vmin=log_l2l_map.min(), vmax=log_l2l_map.max())

            # 2. 提取 L2T (Image-to-Text) 并更新下方柱状图
            l2t_attn = self.attn_matrix[global_token_idx, :self.text_seq_len]
            for bar, h in zip(self.bars, l2t_attn):
                bar.set_height(h)
            
            # 动态调整柱状图 Y 轴
            max_l2t = max(l2t_attn)
            self.ax_bar.set_ylim(0, max_l2t * 1.1 if max_l2t > 0 else 1)
            
            # 找到激活度最高的词
            max_word_idx = np.argmax(l2t_attn)
            max_word = self.tokens[max_word_idx]

            # 隐藏文本上的红框（因为此时是看 L2T，没有选中文本）
            self.bar_highlight.set_alpha(0)

            # 更新标题
            self.ax_img.set_title(f"Image Patch Selected: x={x_tok}, y={y_tok}")
            self.ax_attn.set_title("L2L Attention (Where is this patch looking?)")
            self.ax_bar.set_title(f"L2T Activation | Max Response: '{max_word}' (Idx: {max_word_idx})")

        # ---------------------------------------------------------
        # 场景 B: 点击了下方的【文本柱状图】
        # ---------------------------------------------------------
        elif event.inaxes == self.ax_bar:
            clicked_x = int(round(event.xdata))
            
            if 0 <= clicked_x < self.text_seq_len:
                # 选中的 Text Token 索引
                text_idx = clicked_x
                word = self.tokens[text_idx]

                # --- 新增：提取 T2T (Text-to-Text) 并更新下方柱状图 ---
                t2t_attn = self.attn_matrix[text_idx, :self.text_seq_len]
                for bar, h in zip(self.bars, t2t_attn):
                    bar.set_height(h)
                
                # 动态调整柱状图 Y 轴
                max_t2t = max(t2t_attn)
                self.ax_bar.set_ylim(0, max_t2t * 1.1 if max_t2t > 0 else 1)

                # 更新柱状图红框，使用当前的 T2T 注意力值作为框的高度
                bar_height = t2t_attn[text_idx]
                self.bar_highlight.set_alpha(1)
                # X, Y, Width, Height
                self.bar_highlight.set_bounds(text_idx - 0.4, 0, 0.8, bar_height if bar_height > 0 else 0.1)

                # --- 提取 T2L (Text-to-Image) 并更新右上角热力图 ---
                t2l_attn = self.attn_matrix[text_idx, self.text_seq_len:]
                t2l_map_2d = t2l_attn.reshape((self.h_tok, self.w_tok))
                log_t2l_map = np.log(t2l_map_2d + 1e-6)

                self.im_attn.set_data(log_t2l_map)
                self.im_attn.set_clim(vmin=log_t2l_map.min(), vmax=log_t2l_map.max())

                # 更新标题
                self.ax_attn.set_title(f"T2L Attention (Image areas affected by '{word}')")
                self.ax_bar.set_title(f"T2T Activation | Selected Text Token: '{word}' (Idx: {text_idx})")

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    viewer = InteractiveAttentionViewer(
        pt_path="controller_attention_store.pt",
        batch_idx=0
    )
# %%
