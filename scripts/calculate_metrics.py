import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.metrics import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric


def crop_black_border(image, threshold=5):
    arr = np.array(image)
    mask = arr.mean(axis=2) > threshold

    if not mask.any():
        return image

    ys, xs = np.where(mask)
    top, bottom = ys.min(), ys.max() + 1
    left, right = xs.min(), xs.max() + 1

    return image.crop((left, top, right, bottom))


def load_image(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def find_gt_path(gt_dir, image_id):
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = gt_dir / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--recon-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/metrics")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resize-target", action="store_true")
    parser.add_argument("--lpips-net", type=str, default="vgg")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    recon_dir = Path(args.recon_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    metrics = [
        MSEMetric(name="mse", clip=True),
        PSNRMetric(name="psnr", data_range=1.0, clip=True),
        SSIMMetric(name="ssim", data_range=1.0, clip=True, device=device),
        LPIPSMetric(name="lpips", net=args.lpips_net, clip=True, device=device),
    ]

    recon_paths = sorted(recon_dir.glob("*.png"))

    if len(recon_paths) == 0:
        raise RuntimeError(f"No .png reconstructions found in: {recon_dir}")

    rows = []

    for recon_path in tqdm(recon_paths, desc="Calculating metrics"):
        image_id = recon_path.stem
        gt_path = find_gt_path(gt_dir, image_id)

        if gt_path is None:
            raise FileNotFoundError(f"GT image for '{image_id}' was not found.")

        gt_image = Image.open(gt_path).convert("RGB")
        gt_image = crop_black_border(gt_image)
        gt_tensor = torch.from_numpy(np.asarray(gt_image).astype("float32") / 255.0)
        gt_tensor = gt_tensor.permute(2, 0, 1).unsqueeze(0)

        recon_tensor = load_image(recon_path)

        gt_tensor = gt_tensor.to(device)
        recon_tensor = recon_tensor.to(device)

        if gt_tensor.shape[-2:] != recon_tensor.shape[-2:]:
            if not args.resize_target:
                raise ValueError(
                    "Shape mismatch for '{}': gt={}, recon={}. "
                    "Use --resize-target if this is expected.".format(
                        image_id,
                        tuple(gt_tensor.shape),
                        tuple(recon_tensor.shape),
                    )
                )

            gt_tensor = F.interpolate(
                gt_tensor,
                size=recon_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        row = {"image_id": image_id}

        for metric in metrics:
            value = metric(prediction=recon_tensor, target=gt_tensor)
            row[metric.name] = float(value)

        rows.append(row)

    metric_names = [metric.name for metric in metrics]

    summary = {}
    for name in metric_names:
        values = [row[name] for row in rows]
        summary[name] = float(np.mean(values))

    csv_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w") as f:
        f.write("Metrics summary\n")
        f.write(f"gt_dir: {gt_dir}\n")
        f.write(f"recon_dir: {recon_dir}\n")
        f.write(f"num_images: {len(rows)}\n")
        f.write("\n")

        for name, value in summary.items():
            f.write("{}: {:.6f}\n".format(name, value))

    print()
    print("Metrics summary")
    for name, value in summary.items():
        print("{}: {:.6f}".format(name, value))

    print()
    print(f"Saved per-image metrics to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
