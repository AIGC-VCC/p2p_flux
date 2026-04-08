import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import collections

# --- 配置部分 ---
# 指向你要整理的实验结果目录
RESULT_DIR = Path("output/9-Disney_Style/5_6") 
# 最终拼接图片的保存路径
OUTPUT_GRID_PATH = RESULT_DIR / "hyperparam_grid.png"

# 字体配置 (用于绘制坐标轴标签)
# 尝试加载系统自带的 Arial 字体，如果没有则使用默认字体
try:
    # Linux 常见路径
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" 
    if not os.path.exists(FONT_PATH):
        # macOS 常见路径
        FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    
    label_font = ImageFont.truetype(FONT_PATH, 40)
    title_font = ImageFont.truetype(FONT_PATH, 50)
except math:
    print("Warning: Could not load bold font, using default. Labels might be small.")
    label_font = ImageFont.load_default()
    title_font = ImageFont.load_default()

# 坐标轴区域的预留宽度/高度 (像素)
Y_AXIS_SPACE = 200
X_AXIS_SPACE = 150
# 图片之间的间距
MARGIN = 10 
# 背景颜色 (白色)
BG_COLOR = (255, 255, 255)
# 文字颜色 (黑色)
TEXT_COLOR = (0, 0, 0)

# --- 核心逻辑 ---

def parse_params_from_filename(filename):
    """
    从文件名中解析 end_inject (ei) 和 lambda_shift (ls)。
    例如: output_ei15_ls2.0.png -> (15, 2.0)
    """
    # 使用正则表达式匹配 ei 和 ls 的数值
    pattern = r"output_ei(\d+)_ls([\d.]+)\.png"
    match = re.search(pattern, filename)
    if match:
        ei = int(match.group(1))
        ls = float(match.group(2))
        return ei, ls
    return None


def crop_bottom_right(img_path):
    """
    加载 2x2 的全图，裁剪出右下角 (B') 的部分。
    """
    with Image.open(img_path) as img:
        w, h = img.size
        # FLUX Fill 的 2x2 布局中：
        # 上半部分是 A+A'，下半部分是 B+B'。高度对半开。
        # 左半部分是 A/B，右半部分是 A'/B'。宽度对半开。
        # 注意：这里假设了 img_a 和 img_b 的尺寸是一样的，
        # 这在 run.py 的 build_2x2_input_and_mask 逻辑中是成立的。
        
        left = w // 2
        top = h // 2
        right = w
        bottom = h
        
        # 裁剪出 B'
        cropped_img = img.crop((left, top, right, bottom))
        # 必须 load() 一下，否则 context manager 关闭后图片数据就没了
        cropped_img.load() 
        return cropped_img


def main():
    if not RESULT_DIR.exists():
        print(f"Error: Directory not found at {RESULT_DIR}")
        return

    # 1. 扫描文件并建立映射
    print(f"Scanning files in {RESULT_DIR}...")
    file_map = {} # {(ei, ls): path}
    all_ei = set()
    all_ls = set()

    for file_path in RESULT_DIR.glob("output_ei*_ls*.png"):
        params = parse_params_from_filename(file_path.name)
        if params:
            ei, ls = params
            file_map[(ei, ls)] = file_path
            all_ei.add(ei)
            all_ls.add(ls)

    if not file_map:
        print("No valid output images found.")
        return

    # 2. 排序参数，确定行和列
    # Y轴：end_inject，从上到下递增
    sorted_ei = sorted(list(all_ei))
    # X轴：lambda_shift，从左到右递增
    sorted_ls = sorted(list(all_ls))

    num_rows = len(sorted_ei)
    num_cols = len(sorted_ls)
    print(f"Found grid size: {num_rows} rows (end_inject) x {num_cols} cols (lambda_shift)")

    # 3. 读取第一张图确定单元格尺寸
    first_img_path = list(file_map.values())[0]
    sample_b_prime = crop_bottom_right(first_img_path)
    cell_w, cell_h = sample_b_prime.size
    print(f"Single cell size (B'): {cell_w}x{cell_h}")

    # 4. 计算总画布尺寸
    total_w = Y_AXIS_SPACE + (cell_w + MARGIN) * num_cols
    total_h = X_AXIS_SPACE + (cell_h + MARGIN) * num_rows
    
    # 创建空白画布
    grid_img = Image.new('RGB', (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(grid_img)

    # 5. 绘制坐标轴标题
    # X轴标题 (Lambda Shift)
    x_title = "Lambda Shift"
    tw, th = draw.textbbox((0, 0), x_title, font=title_font)[2:]
    draw.text((Y_AXIS_SPACE + (total_w - Y_AXIS_SPACE - tw) // 2, 20), x_title, fill=TEXT_COLOR, font=title_font)

    # Y轴标题 (End Inject Step)
    # PIL 绘制旋转文字比较麻烦，这里简单处理，分行显示
    y_title_base = "End\nInject\nStep"
    draw.multiline_text((20, X_AXIS_SPACE + 20), y_title_base, fill=TEXT_COLOR, font=title_font, align="center")


    # 6. 开始拼接单元格并绘制刻度标签
    print("Stitching grid...")
    
    # 绘制 X轴 刻度 (lambda_shift)
    for c, ls in enumerate(sorted_ls):
        ls_str = f"{ls:.1f}"
        # 计算文字位置，使其居中于单元格顶部
        tw, th = draw.textbbox((0, 0), ls_str, font=label_font)[2:]
        start_x = Y_AXIS_SPACE + c * (cell_w + MARGIN)
        text_x = start_x + (cell_w - tw) // 2
        text_y = X_AXIS_SPACE - th - 20 # 距离单元格顶部 20px
        draw.text((text_x, text_y), ls_str, fill=TEXT_COLOR, font=label_font)

    for r, ei in enumerate(sorted_ei):
        # 绘制 Y轴 刻度 (end_inject)
        ei_str = str(ei)
        tw, th = draw.textbbox((0, 0), ei_str, font=label_font)[2:]
        start_y = X_AXIS_SPACE + r * (cell_h + MARGIN)
        text_x = Y_AXIS_SPACE - tw - 30 # 距离单元格左侧 30px
        text_y = start_y + (cell_h - th) // 2
        draw.text((text_x, text_y), ei_str, fill=TEXT_COLOR, font=label_font)

        for c, ls in enumerate(sorted_ls):
            # 获取对应的图片路径
            img_path = file_map.get((ei, ls))
            
            current_x = Y_AXIS_SPACE + c * (cell_w + MARGIN)
            current_y = X_AXIS_SPACE + r * (cell_h + MARGIN)

            if img_path:
                try:
                    # 裁剪 B' 并贴到画布上
                    b_prime = crop_bottom_right(img_path)
                    grid_img.paste(b_prime, (current_x, current_y))
                except math:
                    print(f"Warning: Failed to process {img_path}")
            else:
                # 如果某个组合缺了图，画个灰框占位
                draw.rectangle(
                    [current_x, current_y, current_x + cell_w, current_y + cell_h],
                    outline=(200, 200, 200), fill=(240, 240, 240)
                )
                draw.text((current_x + 50, current_y + 50), "Missing", fill=(150, 150, 150), font=label_font)

    # 7. 保存结果
    print(f"Saving grid image to {OUTPUT_GRID_PATH}...")
    grid_img.save(OUTPUT_GRID_PATH)
    print("Done.")

if __name__ == "__main__":
    main()