import argparse
from collections import defaultdict
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets.collate import collate_fn
from src.model.fft_operators import (
    center_crop,
    center_pad,
    fft_convolve,
    normalize_psf_sum,
    psf_to_otf,
)


def forward_chx(target, psf, padded_hw):
    """
    Compute CHx exactly as ADMM forward model, but without psf_gain.

    Args:
        target: tensor [B, C, H, W]
        psf: tensor [B, C, H, W]
        padded_hw: padded size (H_pad, W_pad)

    Returns:
        chx: tensor [B, C, H, W]
    """
    h, w = target.shape[-2:]

    psf = normalize_psf_sum(psf)

    x_padded = center_pad(target, padded_hw)
    otf = psf_to_otf(psf, padded_hw)

    chx_padded = fft_convolve(x_padded, otf)
    chx = center_crop(chx_padded, (h, w))

    return chx


def load_hydra_config(config_name):
    """
    Load Hydra config from src/configs.

    Args:
        config_name: config name without .yaml extension.

    Returns:
        Hydra config.
    """
    config_dir = str(Path("src/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="eval_admm")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-path", type=str, default="outputs/psf_gain.txt")
    parser.add_argument("--print-per-mask", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cfg = load_hydra_config(args.config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    padded_hw = tuple(cfg.model.padded_hw)

    datasets = instantiate(cfg.datasets)

    if args.split == "train":
        dataset = datasets["train"]
    elif args.split in ["val", "test"]:
        dataset = datasets["val"]
    else:
        raise ValueError("Unknown split: {}".format(args.split))

    if args.limit is not None and args.limit > 0:
        dataset._index = dataset._index[: args.limit]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    global_num = 0.0
    global_den = 0.0

    per_mask_num = defaultdict(float)
    per_mask_den = defaultdict(float)
    per_mask_count = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Estimating PSF gain"):
            measurement = batch["measurement"].to(device)
            target = batch["target"].to(device)
            psf = batch["psf"].to(device)
            mask_labels = batch["mask_label"].detach().cpu().tolist()

            chx = forward_chx(target, psf, padded_hw)

            global_num += (measurement * chx).sum().item()
            global_den += (chx * chx).sum().item()

            for i, mask_label in enumerate(mask_labels):
                mask_label = int(mask_label)

                b_i = measurement[i : i + 1]
                chx_i = chx[i : i + 1]

                per_mask_num[mask_label] += (b_i * chx_i).sum().item()
                per_mask_den[mask_label] += (chx_i * chx_i).sum().item()
                per_mask_count[mask_label] += 1

    global_gain = global_num / max(global_den, 1e-12)

    per_mask_gains = {}
    for mask_label in sorted(per_mask_num):
        per_mask_gains[mask_label] = per_mask_num[mask_label] / max(
            per_mask_den[mask_label],
            1e-12,
        )

    gains = torch.tensor(list(per_mask_gains.values()), dtype=torch.float32)

    per_mask_mean = gains.mean().item()
    per_mask_std = gains.std(unbiased=False).item()
    per_mask_cv = per_mask_std / max(per_mask_mean, 1e-12)

    print()
    print("PSF gain estimation")
    print("split: {}".format(args.split))
    print("samples: {}".format(len(dataset)))
    print("batch_size: {}".format(args.batch_size))
    print("padded_hw: {}".format(padded_hw))
    print()
    print("global_gain: {:.8f}".format(global_gain))
    print("per_mask_mean: {:.8f}".format(per_mask_mean))
    print("per_mask_std: {:.8f}".format(per_mask_std))
    print("per_mask_cv: {:.8f}".format(per_mask_cv))

    if args.print_per_mask:
        print()
        print("per-mask gains:")
        for mask_label, gain in per_mask_gains.items():
            print(
                "mask_{}: gain={:.8f}, count={}".format(
                    mask_label,
                    gain,
                    per_mask_count[mask_label],
                )
            )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("PSF gain estimation\n")
        f.write("split: {}\n".format(args.split))
        f.write("samples: {}\n".format(len(dataset)))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("padded_hw: {}\n".format(padded_hw))
        f.write("\n")
        f.write("global_gain: {:.8f}\n".format(global_gain))
        f.write("per_mask_mean: {:.8f}\n".format(per_mask_mean))
        f.write("per_mask_std: {:.8f}\n".format(per_mask_std))
        f.write("per_mask_cv: {:.8f}\n".format(per_mask_cv))
        f.write("\n")
        f.write("per-mask gains:\n")
        for mask_label, gain in per_mask_gains.items():
            f.write(
                "mask_{}: gain={:.8f}, count={}\n".format(
                    mask_label,
                    gain,
                    per_mask_count[mask_label],
                )
            )

    print()
    print("Saved detailed results to: {}".format(out_path))


if __name__ == "__main__":
    main()
