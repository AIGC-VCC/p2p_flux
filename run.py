import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T
import numpy as np
from diffusers import FluxFillPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel

from diffusers.utils import load_image
from attn_proc.vanilla import VanillaFluxAttnProcessor
from attn_proc.sac import SACFluxAttnProcessor
from attn_proc.feature_align import FeatureAlignFluxAttnProcessor, FeatureAlignConfig

task_list = [
    "9-Disney_Style",
    "6-Animal_Airforce",
]

Relation252K_dataset = "~/.cache/huggingface/hub/datasets--handsomeWilliam--Relation252K/snapshots/77d3267468d11b7625671f123f57d09424a58631"

def get_sample(dataset_dir, task, idx):
    final_dir = Path(os.path.expanduser(dataset_dir)) / task / "Group1"
    labels_path = final_dir / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    if idx <= 0 or idx > len(labels_data):
        raise IndexError(f"Index {idx} out of range for task {task} with only {len(labels_data)} samples.")
    label = labels_data[idx-1].copy()
    img_path = final_dir / label['img_name']
    label['img_path'] = str(img_path)
    return label


def get_sample_pair(dataset_dir, task, idx_a, idx_b):
    sample_a = get_sample(dataset_dir, task, idx_a)
    sample_b = get_sample(dataset_dir, task, idx_b)
    combined_sample = {
        'task': task,
        'samples': [sample_a, sample_b]
    }
    return combined_sample


def get_output_dir(base_dir, task, idx_a, idx_b):
    output_dir = Path(base_dir) / f"{task}" / f"{idx_a}_{idx_b}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_2x2_input_and_mask(img_a, img_b, mask_strategy='gray'):
    b_w, b_h = img_b.size
    half_b_w = b_w // 2

    img_b_left = img_b.crop((0, 0, half_b_w, b_h))
    row_b_input = Image.new('RGB', (b_w, b_h))
    row_b_input.paste(img_b_left, (0, 0))

    if mask_strategy == 'gray':
        gray_block = Image.new('RGB', (b_w - half_b_w, b_h), (128, 128, 128))
        row_b_input.paste(gray_block, (half_b_w, 0))
    elif mask_strategy == 'copy':
        row_b_input.paste(img_b_left, (half_b_w, 0))
    else:
        raise ValueError(f"Unsupported mask_strategy: {mask_strategy}. Use 'gray' or 'copy'.")

    total_width = max(img_a.width, b_w)
    total_height = img_a.height + b_h

    input_image = Image.new('RGB', (total_width, total_height))
    input_image.paste(img_a, (0, 0))
    input_image.paste(row_b_input, (0, img_a.height))

    mask_image = Image.new('L', (total_width, total_height), 0)
    mask_block = Image.new('L', (b_w - half_b_w, b_h), 255)
    mask_image.paste(mask_block, (half_b_w, img_a.height))

    gt_image = Image.new('RGB', (total_width, total_height))
    gt_image.paste(img_a, (0, 0))
    gt_image.paste(img_b, (0, img_a.height))

    return input_image, mask_image, gt_image


def generate_2x2_input_mask_gt(pair, mask_strategy='gray'):
    sample_a, sample_b = pair['samples']

    img_a = Image.open(sample_a['img_path']).convert('RGB')
    img_b = Image.open(sample_b['img_path']).convert('RGB')

    input_image, mask_image, gt_image = build_2x2_input_and_mask(img_a, img_b, mask_strategy=mask_strategy)

    return input_image, mask_image, gt_image


def generate_2x2_prompt(pair):
    sample_a, sample_b = pair['samples']
    return (
        "A 2x2 layout of images.\n"
        f"[left-top] {sample_a['left_image_description']}\n"
        f"[right-top] {sample_a['right_image_description']}\n"
        f"[left-bottom] {sample_b['left_image_description']}\n"
        f"[right-bottom] {sample_b['right_image_description']}"
    )


def load_flux_to_2_gpu(model_path, gpu0, gpu1, dtype=torch.bfloat16):
    custom_device_map = {
        "pos_embed": gpu0,
        "time_text_embed": gpu0,
        "context_embedder": gpu0,
        "x_embedder": gpu0,
        "transformer_blocks": gpu0,
        # "single_transformer_blocks": gpu1,
        "norm_out": gpu1,
        "proj_out": gpu1
    }
    for i in range(19):
        custom_device_map[f"single_transformer_blocks.{i}"] = gpu0
    for i in range(19, 38):
        custom_device_map[f"single_transformer_blocks.{i}"] = gpu1

    transformer = FluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map=custom_device_map,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=dtype).to(gpu1)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).to(gpu1)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(gpu1)
    pipe = FluxFillPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=dtype,
    )

    return pipe


def save_attention_map(tokenizer_2, prompt, max_sequence_length, image, attn_processor, out_width, out_height, attention_store_path="controller_attention_store.pt"):
    to_tensor = T.ToTensor()
    # 1. 提取 Prompt 对应的 Token IDs (FLUX 用 tokenizer_2 处理)
    text_inputs = tokenizer_2(
        prompt, 
        padding="max_length", 
        max_length=max_sequence_length, 
        truncation=True, 
        return_tensors="pt"
    )
    token_ids = text_inputs.input_ids[0].tolist()
    # 2. 将 IDs 转换回可读的单词/子词
    decoded_tokens = pipe.tokenizer_2.convert_ids_to_tokens(token_ids)

    print(f"attn map shape: {attn_processor.attention_store.shape}, take up memory: {attn_processor.attention_store.element_size() * attn_processor.attention_store.nelement() / (1024**2):.2f} MB")
    save_dict = {
        "image": torch.stack([to_tensor(img) for img in image]),
        "attention_map": attn_processor.attention_store.to("cpu") / attn_processor.attn_save_counter,  # 平均注意力图
        "tokens": decoded_tokens,  # <--- 把文本 token 列表存进去！
        "seq_len": attn_processor.seq_len, # 把长度也存下来，方便切片
        "text_seq_len": attn_processor.text_seq_len,
        "latent_seq_len": attn_processor.latent_seq_len,
        "out_width": out_width,
        "out_height": out_height,
    }
    torch.save(save_dict, attention_store_path)


if __name__ == "__main__":
    task = "9-Disney_Style"
    idx_a = 1
    idx_b = 2
    pair = get_sample_pair(Relation252K_dataset, task, idx_a, idx_b)
    output_dir = get_output_dir("output", task, idx_a, idx_b)

    input_image, mask_image, gt_image = generate_2x2_input_mask_gt(pair, mask_strategy='gray')
    prompt = generate_2x2_prompt(pair)

    model_path = "/home/frain/Documents/FLUX.1-Fill"
    pipe = load_flux_to_2_gpu(model_path, gpu0="cuda:0", gpu1="cuda:1", dtype=torch.bfloat16)

    num_inference_steps = 50
    max_sequence_length = 256
    # out_width same as input width, out_height same as input height
    out_width = input_image.width
    out_height = input_image.height
    exp_config = FeatureAlignConfig(
        start_inject=0,
        end_inject=5,
        lambda_shift=2.0,
        use_soft_mask=False,
    )
    attn_processor = FeatureAlignFluxAttnProcessor(pipe, prompt, out_width, out_height, config=exp_config)

    images = pipe(
        prompt=prompt,
        image=input_image,
        mask_image=mask_image,
        width=out_width,
        height=out_height,
        guidance_scale=30,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(22)
    ).images

    # save output image
    images[0].save(output_dir / "output.png")
    gt_image.save(output_dir / "gt.png")

