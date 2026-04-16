import os
import json
import subprocess
import itertools
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

task_list = [
    "7-SpotRemoval",
    "8-Character_Thinner",
    "WatercolorHandpainted-Animal1"

    "107-Strokes-LOGO",
    "87-Gesture_Edit",
    "82-Pokemon_Evolution",
    "80-DipPenArt",
]

# 借用 run.py 里的工具函数进行前置数据准备
from run import get_sample_pair, generate_2x2_prompt, generate_2x2_input_mask_gt, Relation252K_dataset

def setup_workdir(task, idx_a, idx_b):
    workdir = Path("output") / task / f"{idx_a}_{idx_b}"
    workdir.mkdir(parents=True, exist_ok=True)
    
    config_path = workdir / "config.json"

    # 🌟 1. 尝试读取已存在的配置
    existing_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {config_path} is corrupted, will recreate it.")

    # 获取基础信息
    pair = get_sample_pair(Relation252K_dataset, task, idx_a, idx_b)
    
    # 默认路径
    default_input_path = str(workdir / "input_image.png")
    default_mask_path = str(workdir / "mask_image.png")
    default_gt_path = str(workdir / "gt.png")

    # 🌟 2. 构建基础默认配置
    base_config = {
        "task": task,
        "idx_a": idx_a,
        "idx_b": idx_b,
        "prompt": pair["samples"][1]['right_image_description'],
        "example_img_path": pair['samples'][0]['img_path'],
        "tobeedit_img_path": pair['samples'][1]['img_path'],
        "input_img_path": default_input_path,
        "mask_img_path": default_mask_path,
        "gt_img_path": default_gt_path
    }

    # 🌟 3. 合并配置：用手动修改过的现有配置覆盖默认配置
    for key, value in existing_config.items():
        base_config[key] = value

    # 🌟 4. 图片的按需生成（防覆盖）
    # 如果配置文件里指定的 input_img_path 文件确实存在，说明图片已经准备好了，跳过生成
    if not (Path(base_config["input_img_path"]).exists() and Path(base_config["mask_img_path"]).exists()):
        print(f"Generating images for {task} {idx_a}_{idx_b}...")
        input_img, mask_img, gt_img = generate_2x2_input_mask_gt(pair, mask_strategy='gray')
        input_img.save(base_config["input_img_path"])
        mask_img.save(base_config["mask_img_path"])
        gt_img.save(base_config["gt_img_path"])
    else:
        print(f"Input images already exist for {task} {idx_a}_{idx_b}, skipping generation.")

    # 🌟 5. 将最终的完整配置写回
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(base_config, f, indent=4, ensure_ascii=False)
        
    return str(workdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="9-Disney_Style")
    parser.add_argument("--idx_a", type=int, default=6)
    parser.add_argument("--idx_b", type=int, default=5)
    args = parser.parse_args()

    # 1. 准备实验目录和静态数据
    print(f"Setting up workspace for task: {args.task}, pair: {args.idx_a}_{args.idx_b}")
    workdir = setup_workdir(args.task, args.idx_a, args.idx_b)

    # 2. 初始化 GPU 队列
    gpu_pairs = [("0", "1"), ("2", "3"), ("4", "5"), ("6", "7")]
    gpu_queue = Queue()
    for pair in gpu_pairs:
        gpu_queue.put(pair)

    # 3. 定义参数空间并生成任务
    end_inject_list = [-1, 0, 3, 5]
    lambda_shift_list = [-5, -2.0, 0]
    tasks = list(itertools.product(end_inject_list, lambda_shift_list))

    def run_experiment(task_params):
        ei, ls = task_params
        gpu0, gpu1 = gpu_queue.get()
        
        print(f"Starting end_step={ei}, tau_b={ls} on GPUs {gpu0},{gpu1}")
        
        # 传递 workdir 而非单独的 task/idx
        cmd = [
            "python", "run.py",
            "--workdir", workdir,
            "--tau_a", str(ei),
            "--tau_b", str(ls),
            "--gpu0", f"cuda:0",
            "--gpu1", f"cuda:1"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f"{gpu0},{gpu1}"
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError:
            print(f"Task tau_a={ei}, tau_b={ls} failed!")
        finally:
            gpu_queue.put((gpu0, gpu1))
            print(f"Finished tau_a={ei}, tau_b={ls}")

    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_experiment, tasks)

    # 5. 生成汇总网格图
    print(f"All runs finished. Generating grid for {workdir}...")
    subprocess.run(["python", "make_grid.py", "--result_dir", workdir], check=True)

if __name__ == "__main__":
    main()