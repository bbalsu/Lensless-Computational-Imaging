import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def crop_black_border(img, threshold=5):
    arr = np.array(img)
    mask = arr.mean(axis=2) > threshold

    if not mask.any():
        return img

    ys, xs = np.where(mask)
    top, bottom = ys.min(), ys.max() + 1
    left, right = xs.min(), xs.max() + 1

    return img.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--recon-dir", type=str, required=True)
    parser.add_argument(
        "--out-path", type=str, default="outputs/demo_visualization.png"
    )
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--crop-gt-black-border", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    recon_dir = Path(args.recon_dir)
    lensless_dir = data_dir / "lensless"
    lensed_dir = data_dir / "lensed"

    image_ids = sorted([p.stem for p in lensless_dir.glob("*.png")])
    image_ids = [
        image_id for image_id in image_ids if (recon_dir / f"{image_id}.png").exists()
    ]
    image_ids = image_ids[: args.n_images]

    if len(image_ids) == 0:
        raise RuntimeError("No matching lensless/reconstruction image ids found.")

    has_lensed = lensed_dir.exists()
    n_cols = 3 if has_lensed else 2

    plt.figure(figsize=(4 * n_cols, 4 * len(image_ids)))

    for row, image_id in enumerate(image_ids):
        lensless = Image.open(lensless_dir / f"{image_id}.png").convert("RGB")
        recon = Image.open(recon_dir / f"{image_id}.png").convert("RGB")

        col = 1

        plt.subplot(len(image_ids), n_cols, row * n_cols + col)
        plt.imshow(lensless)
        plt.title(f"{image_id}: lensless")
        plt.axis("off")
        col += 1

        if has_lensed and (lensed_dir / f"{image_id}.png").exists():
            lensed = Image.open(lensed_dir / f"{image_id}.png").convert("RGB")
            if args.crop_gt_black_border:
                lensed = crop_black_border(lensed)

            plt.subplot(len(image_ids), n_cols, row * n_cols + col)
            plt.imshow(lensed)
            plt.title("lensed / GT")
            plt.axis("off")
            col += 1

        plt.subplot(len(image_ids), n_cols, row * n_cols + col)
        plt.imshow(recon)
        plt.title("reconstruction")
        plt.axis("off")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
