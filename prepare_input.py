import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image

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


def generate_2x2_input_and_mask(dataset_dir, task, idx_a, idx_b, output_base='output', mask_strategy='gray'):
    pair = get_sample_pair(dataset_dir, task, idx_a, idx_b)
    sample_a, sample_b = pair['samples']

    img_a = Image.open(sample_a['img_path']).convert('RGB')
    img_b = Image.open(sample_b['img_path']).convert('RGB')

    input_image, mask_image, gt_image = build_2x2_input_and_mask(img_a, img_b, mask_strategy=mask_strategy)

    output_dir = get_output_dir(output_base, task, idx_a, idx_b)
    img_a.save(output_dir / "a.png")
    img_b.save(output_dir / "b.png")
    input_image.save(output_dir / "input.png")
    mask_image.save(output_dir / "mask.png")
    gt_image.save(output_dir / "gt.png")

    prompt = (
        "A 2x2 layout of images.\n"
        f"[left-top] {sample_a['left_image_description']}\n"
        f"[right-top] {sample_a['right_image_description']}\n"
        f"[left-bottom] {sample_b['left_image_description']}\n"
        f"[right-bottom] {sample_b['right_image_description']}"
    )
    with open(output_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    return {
        "task": task,
        "idx_a": idx_a,
        "idx_b": idx_b,
        "output_dir": str(output_dir),
        "input_path": str(output_dir / "input.png"),
        "mask_path": str(output_dir / "mask.png"),
        "gt_path": str(output_dir / "gt.png"),
        "prompt_path": str(output_dir / "prompt.txt")
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2x2 input image and mask by given idx_a/idx_b.")
    parser.add_argument("--dataset_dir", type=str, default=Relation252K_dataset)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--idx_a", type=int, required=True)
    parser.add_argument("--idx_b", type=int, required=True)
    parser.add_argument("--output_base", type=str, default="output")
    parser.add_argument("--mask_strategy", type=str, default="gray", choices=["gray", "copy"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = generate_2x2_input_and_mask(
        dataset_dir=args.dataset_dir,
        task=args.task,
        idx_a=args.idx_a,
        idx_b=args.idx_b,
        output_base=args.output_base,
        mask_strategy=args.mask_strategy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

"""
ds = Realtion252KDataset("~/.cache/huggingface/hub/datasets--handsomeWilliam--Relation252K/snapshots/77d3267468d11b7625671f123f57d09424a58631")
print(f"总共找到 {len(ds.get_task_list())} 个任务。")
print("所有任务名称：")
for task in ds.get_task_list():
    print(f" - {task.task_name}")
print("task 0 的前 2 个样本：")
for sample in ds.get_task_list()[134].get_samples()[:2]:
    print(sample)

def get_or_generate_vicl_data(base_dir, batch_id, mask_strategy='gray'):
    batch_dir = Path(f"batch/{batch_id}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(os.path.expanduser(base_dir))

    info_file = batch_dir / "info.json"

    # 1. 根据 info.json 确定样本 (恢复或随机抽取)
    if info_file.exists():
        print(f"📄 发现 info.json，按已有记录恢复样本数据...")
        with open(info_file, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        
        task_name = info_data["task"]
        group_name = info_data["group"]
        sample_a = info_data["sample_a"]
        sample_b = info_data["sample_b"]
        # 优先使用最初生成时记录的 mask 策略，保持一致性
        mask_strategy = info_data.get("mask_strategy", mask_strategy) 
        selected_group = base_path / task_name / group_name
    else:
        print(f"✨ 未找到 info.json，开始随机抽取新数据...")
        task_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name != ".git"]
        if not task_dirs:
            raise ValueError("未找到任何任务文件夹，请检查路径。")

        selected_task = random.choice(task_dirs)
        group_dirs = [d for d in selected_task.iterdir() if d.is_dir()]
        if not group_dirs:
            raise ValueError(f"任务 {selected_task.name} 下未找到 Group 文件夹。")
            
        selected_group = random.choice(group_dirs)
        labels_path = selected_group / "labels.json"
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

        if len(labels_data) < 2:
            raise ValueError(f"Group {selected_group.name} 中的数据少于两条，无法构成 Pair。")

        sample_a, sample_b = random.sample(labels_data, 2)

        # 保存 info.json，作为该批次的“唯一事实来源”
        info_dict = {
            "task": selected_task.name,
            "group": selected_group.name,
            "sample_a": sample_a,
            "sample_b": sample_b,
            "mask_strategy": mask_strategy
        }
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info_dict, f, indent=4, ensure_ascii=False)

    # 2. 检查并处理 prompt.txt
    prompt_file = batch_dir / "prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    else:
        print("📝 prompt.txt 缺失，正在重新生成...")
        prompt = (
            "A 2x2 layout of images.\n"
            f"[left-top] {sample_a['left_image_description']}\n"
            f"[right-top] {sample_a['right_image_description']}\n"
            f"[left-bottom] {sample_b['left_image_description']}\n"
            f"[right-bottom] {sample_b['right_image_description']}"
        )
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

    # 3. 检查缺失的图像文件并按需生成
    required_images = ["a.png", "b.png", "input.png", "mask.png", "gt.png"]
    missing_images = [img for img in required_images if not (batch_dir / img).exists()]

    if missing_images:
        print(f"🖼️ 发现缺失图像: {missing_images}，正在根据 info.json 重新构建...")
        img_a_path = selected_group / sample_a['img_name']
        img_b_path = selected_group / sample_b['img_name']

        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
        
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

        # 仅保存缺失的文件，绝对不覆盖已存在的文件
        if "a.png" in missing_images: img_a.save(batch_dir / "a.png")
        if "b.png" in missing_images: img_b.save(batch_dir / "b.png")
        if "input.png" in missing_images: input_image.save(batch_dir / "input.png")
        if "mask.png" in missing_images: mask_image.save(batch_dir / "mask.png")
        if "gt.png" in missing_images: gt_image.save(batch_dir / "gt.png")
    
    # 4. 从本地读取最新的 input 和 mask
    # 这样做可以确保：如果你用 PS 手动编辑并覆盖了 mask.png，模型也会读到你编辑后的版本
    final_input = Image.open(batch_dir / "input.png").convert('RGB')
    final_mask = Image.open(batch_dir / "mask.png").convert('L')

    return final_input, final_mask, prompt, batch_dir

"""