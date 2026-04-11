import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

import torch
import torchvision.transforms as T
import numpy as np
from diffusers import FluxFillPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel

from attn_proc.sac import SACFluxAttnProcessor

Relation252K_dataset = "~/.cache/huggingface/hub/datasets--handsomeWilliam--Relation252K/snapshots/77d3267468d11b7625671f123f57d09424a58631"

def get_sample(dataset_dir, task, idx):
    task_dir = Path(os.path.expanduser(dataset_dir)) / task
    final_dir = sorted(d for d in task_dir.iterdir() if d.is_dir())[0]
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

    assert img_a.size == img_b.size, f"Expected img a and b to have the same size, but got {img_a.size} and {img_b.size}"

    input_image, mask_image, gt_image = build_2x2_input_and_mask(img_a, img_b, mask_strategy=mask_strategy)

    return input_image, mask_image, gt_image


def split_1x2_image(input_image):
    w, h = input_image.size
    half_w = w // 2
    assert half_w * 2 == w, "Image width must be even to split into two equal halves."
    left_half = input_image.crop((0, 0, half_w, h))
    right_half = input_image.crop((half_w, 0, w, h))
    return left_half, right_half


def concat_to_1x2_image(left_half, right_half):
    w, h = left_half.size
    total_width = w * 2
    total_height = h
    new_image = Image.new('RGB', (total_width, total_height))
    new_image.paste(left_half, (0, 0))
    new_image.paste(right_half, (w, 0))
    return new_image


def generate_1x2_input_mask_gt(pair, mask_strategy='gray'):
    sample_a, sample_b = pair['samples']

    img_a = Image.open(sample_a['img_path']).convert('RGB')
    img_b = Image.open(sample_b['img_path']).convert('RGB')

    left_half_a, right_half_a = split_1x2_image(img_a)
    left_half_b, right_half_b = split_1x2_image(img_b)

    input_image_a = concat_to_1x2_image(left_half_a, left_half_b)
    input_image_b = concat_to_1x2_image(right_half_a, right_half_b)

    if mask_strategy == 'gray':
        mask_block = Image.new('L', (input_image_b.width // 2, input_image_b.height), 255)
        mask_image_b = Image.new('L', input_image_b.size, 0)
        mask_image_b.paste(mask_block, (input_image_b.width // 2, 0))
    elif mask_strategy == 'copy':
        mask_image_b = Image.new('L', input_image_b.size, 0)
        mask_image_b.paste(Image.new('L', (input_image_b.width // 2, input_image_b.height), 255), (input_image_b.width // 2, 0))
    else:
        raise ValueError(f"Unsupported mask_strategy: {mask_strategy}. Use 'gray' or 'copy'.")

    gt_image_a = concat_to_1x2_image(left_half_a, left_half_b)
    gt_image_b = concat_to_1x2_image(right_half_a, right_half_b)

    return [input_image_a, input_image_b], [Image.new('L', input_image_a.size, 0), mask_image_b], [gt_image_a, gt_image_b]


def generate_2x2_prompt(pair):
    sample_a, sample_b = pair['samples']
    return (
        "A 2x2 layout of images.\n"
        f"[left-top] {sample_a['left_image_description']}\n"
        f"[right-top] {sample_a['right_image_description']}\n"
        f"[left-bottom] {sample_b['left_image_description']}\n"
        f"[right-bottom] {sample_b['right_image_description']}"
    )


def generate_1x2_2batch_prompt(pair):
    sample_a, sample_b = pair['samples']
    return [(
        "A 1x2 layout of images.\n"
        f"[left] {sample_a['left_image_description']}"
        f"[right] {sample_b['left_image_description']}"
    ), (
        "A 1x2 layout of images.\n"
        f"[left] {sample_a['right_image_description']}"
        f"[right] {sample_b['right_image_description']}"
    )]


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
    decoded_tokens = tokenizer_2.convert_ids_to_tokens(token_ids)

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
    parser = argparse.ArgumentParser()
    # 核心变动：接受 workdir 作为入口
    parser.add_argument("--workdir", type=str, required=True, help="Path to the experiment working directory")
    parser.add_argument("--end_step", type=int, default=8, help="End step for SAC injection")
    parser.add_argument("--tau_a", type=float, default=1.6, help="Logit alignment tau for A' attending to A")
    parser.add_argument("--tau_b", type=float, default=1.0, help="Logit alignment tau for B' attending to B")
    parser.add_argument("--gpu0", type=str, default="cuda:0")
    parser.add_argument("--gpu1", type=str, default="cuda:1")
    parser.add_argument("--savemap", action="store_true", help="Save attention map after inference")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    config_path = workdir / "config.json"
    
    # 1. 加载配置和静态图片
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    prompt = config["prompt"]
    input_image = Image.open(config["input_img_path"]).convert("RGB")
    mask_image = Image.open(config["mask_img_path"]).convert("L")
    
    # 2. 初始化模型
    model_path = "/home/frain/Documents/FLUX.1-Fill"
    pipe = load_flux_to_2_gpu(model_path, gpu0=args.gpu0, gpu1=args.gpu1, dtype=torch.bfloat16)

    out_width, out_height = input_image.width, input_image.height
    num_inference_steps = 50
    max_sequence_length = 256
    
    attn_processor = SACFluxAttnProcessor(
        pipe=pipe, 
        prompt=prompt, 
        out_width=out_width, 
        out_height=out_height, 
        end_step=args.end_step,
        b_a_copyto_bp_ap_tau=args.tau_a,
        b_b_copyto_bp_b_tau=args.tau_b
    )

    # 3. 推理
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

    # 4. 注入 Metadata 并保存
    metadata = PngInfo()
    metadata.add_text("end_step", str(args.end_step))
    metadata.add_text("tau_a", str(args.tau_a))
    metadata.add_text("tau_b", str(args.tau_b))
    metadata.add_text("prompt", prompt)
    
    file_name = f"output_ei{args.tau_a}_ls{args.tau_b}.png"
    out_path = workdir / file_name
    images[0].save(out_path, pnginfo=metadata)

    # 5. 保存注意力图（如果指定）
    if args.savemap:
        attention_store_path = workdir / f"attention_map_ei{args.tau_a}_ls{args.tau_b}.pt"
        save_attention_map(
            pipe.tokenizer_2,
            prompt,
            max_sequence_length,
            images,
            attn_processor,
            out_width,
            out_height,
            # str(attention_store_path)
        )

    # 6. 记录到全局日志中
    log_file = Path("output") / "experiments.jsonl"
    log_data = {
        "task": config["task"],
        "idx_a": config["idx_a"],
        "idx_b": config["idx_b"],
        "end_step": args.end_step,
        "tau_a": args.tau_a,
        "tau_b": args.tau_b,
        "workdir": str(workdir),
        "output_path": str(out_path)
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")