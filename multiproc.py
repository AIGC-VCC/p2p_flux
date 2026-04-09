import os
import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# 你的 8 张 4090，两两分组，支持 4 个并行任务
gpu_pairs = [("0", "1"), ("2", "3"), ("4", "5"), ("6", "7")]
gpu_queue = Queue()
for pair in gpu_pairs:
    gpu_queue.put(pair)

# 定义参数空间
end_inject_list = [-1.0, -0.5, 0, 0.5, 1.0]
lambda_shift_list = [-1.0, -0.5, 0, 0.5, 2.0]

# 生成所有组合
tasks = list(itertools.product(end_inject_list, lambda_shift_list))

def run_experiment(task):
    ei, ls = task
    # 从队列中获取可用的 GPU 对
    gpu0, gpu1 = gpu_queue.get()
    
    print(f"Starting end_inject={ei}, lambda_shift={ls} on GPUs {gpu0},{gpu1}")
    
    # 构建执行命令
    cmd = [
        "python", "run.py",
        "--tau_a", str(ei),
        "--tau_b", str(ls),
        "--gpu0", f"cuda:0",
        "--gpu1", f"cuda:1"
    ]
    
    # 隔离可见 GPU（可选，双保险）
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{gpu0},{gpu1}"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Task ei={ei}, ls={ls} failed!")
    finally:
        # 任务结束，归还 GPU 对
        gpu_queue.put((gpu0, gpu1))
        print(f"Finished end_inject={ei}, lambda_shift={ls}")

if __name__ == "__main__":
    # 使用 4 个 worker 跑满 4 个进程
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_experiment, tasks)