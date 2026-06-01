import argparse
import csv
import math
import re
from pathlib import Path
from PIL import Image, ImageFilter


OUTPUT_RE = re.compile(r"output_ei(?P<tau_a>[\d.-]+)_ls(?P<tau_b>[\d.-]+)\.png$")


def load_rgb(path):
    return Image.open(path).convert("RGB")


def crop_quadrant(img, row, col):
    w, h = img.size
    half_w = w // 2
    half_h = h // 2
    return img.crop((col * half_w, row * half_h, (col + 1) * half_w, (row + 1) * half_h))


def to_gray(img, size=None):
    if size is not None:
        img = img.resize(size, Image.Resampling.BICUBIC)
    return img.convert("L")


def gray_values(gray):
    return list(gray.getdata())


def otsu_threshold_255(values):
    hist = [0] * 256
    for value in values:
        hist[int(value)] += 1
    total = sum(hist)
    if total == 0:
        return 128

    sum_total = sum(i * count for i, count in enumerate(hist))
    weight_bg = 0
    sum_bg = 0
    best_threshold = 128
    best_variance = -1.0
    for i, count in enumerate(hist):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * count
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = i
    return best_threshold


def foreground_mask(gray):
    ink_values = [255 - value for value in gray_values(gray)]
    threshold = max(otsu_threshold_255(ink_values), 15)
    return [value > threshold for value in ink_values]


def sobel_edges(gray):
    edges = gray.filter(ImageFilter.FIND_EDGES)
    values = gray_values(edges)
    if not values:
        return []
    sorted_values = sorted(values)
    percentile_threshold = sorted_values[int(0.88 * (len(sorted_values) - 1))]
    threshold = max(percentile_threshold, otsu_threshold_255(values))
    return [value > threshold for value in values]


def binary_f1(pred, target):
    tp = fp = fn = 0
    for pred_value, target_value in zip(pred, target):
        if pred_value and target_value:
            tp += 1
        elif pred_value and not target_value:
            fp += 1
        elif not pred_value and target_value:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def binary_iou(a, b):
    inter = union = 0
    for a_value, b_value in zip(a, b):
        if a_value or b_value:
            union += 1
            if a_value and b_value:
                inter += 1
    return inter / max(union, 1)


def evaluate_generated(generated_b_prime, target_original_b):
    size = generated_b_prime.size
    gen_gray = to_gray(generated_b_prime)
    target_gray = to_gray(target_original_b, size=size)

    edge_f1 = binary_f1(sobel_edges(gen_gray), sobel_edges(target_gray))
    fg_iou = binary_iou(foreground_mask(gen_gray), foreground_mask(target_gray))
    return 0.65 * edge_f1 + 0.35 * fg_iou


def parse_output(path):
    if path.name == "output_no_sac.png":
        return "no_sac", None, None
    match = OUTPUT_RE.match(path.name)
    if not match:
        return None
    return "sac", float(match.group("tau_a")), float(match.group("tau_b"))


def format_optional_float(value):
    if value is None:
        return ""
    return f"{value:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing output images and gt.png")
    parser.add_argument("--csv_name", type=str, default="metrics.csv")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    gt_path = result_dir / "gt.png"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing gt image: {gt_path}")

    gt = load_rgb(gt_path)
    target_original_b = crop_quadrant(gt, row=1, col=0)

    rows = []
    for path in sorted(result_dir.glob("output*.png")):
        parsed = parse_output(path)
        if parsed is None:
            continue
        run_type, tau_a, tau_b = parsed
        generated = load_rgb(path)
        generated_b_prime = crop_quadrant(generated, row=1, col=1)
        restoration_score = evaluate_generated(generated_b_prime, target_original_b)
        rows.append(
            {
                "file": path.name,
                "run_type": run_type,
                "tau_a": format_optional_float(tau_a),
                "tau_b": format_optional_float(tau_b),
                "restoration_score": f"{restoration_score:.6f}",
            }
        )

    if not rows:
        raise FileNotFoundError(f"No output images found in {result_dir}")

    def sort_key(row):
        if row["run_type"] == "no_sac":
            return (0, -math.inf, -math.inf)
        return (1, float(row["tau_a"]), float(row["tau_b"]))

    rows.sort(key=sort_key)
    csv_path = result_dir / args.csv_name
    fieldnames = [
        "file",
        "run_type",
        "tau_a",
        "tau_b",
        "restoration_score",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} metric rows to {csv_path}")


if __name__ == "__main__":
    main()
