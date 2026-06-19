import time
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils.checkpoint_utils import load_model_checkpoint
from src.utils.eval_utils import (
    build_writer,
    get_dataset_by_split,
    limit_dataset,
    make_loader,
    resolve_device,
    synchronize_if_needed,
)


def load_model_config(model_config_name):
    model_config_path = Path("src/configs/model") / "{}.yaml".format(model_config_name)

    if not model_config_path.exists():
        raise FileNotFoundError(
            "Model config does not exist: {}".format(model_config_path)
        )

    return OmegaConf.load(model_config_path)


def benchmark_model(model, loader, device, warmup_batches):
    model.eval()

    total_images = 0
    total_time_sec = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Benchmarking")):
            measurement = batch["measurement"].to(device)
            psf = batch["psf"].to(device)

            if batch_idx < warmup_batches:
                _ = model(measurement, psf)
                synchronize_if_needed(device)
                continue

            synchronize_if_needed(device)
            start_time = time.perf_counter()

            _ = model(measurement, psf)

            synchronize_if_needed(device)
            end_time = time.perf_counter()

            batch_time_sec = end_time - start_time
            batch_size = measurement.shape[0]

            total_time_sec += batch_time_sec
            total_images += batch_size

    sec_per_image = total_time_sec / max(total_images, 1)
    ms_per_image = sec_per_image * 1000.0
    images_per_sec = total_images / max(total_time_sec, 1e-12)

    return {
        "total_images": total_images,
        "total_time_sec": total_time_sec,
        "sec_per_image": sec_per_image,
        "ms_per_image": ms_per_image,
        "images_per_sec": images_per_sec,
    }


def run_single_speed_benchmark(
    method_cfg,
    loader,
    benchmark_cfg,
    device,
):
    method_name = method_cfg.name

    print()
    print("=" * 80)
    print("Benchmarking method: {}".format(method_name))
    print("=" * 80)

    model_cfg = load_model_config(method_cfg.model_config)
    model = instantiate(model_cfg).to(device)

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=method_cfg.checkpoint_path,
        device=device,
        strict=True,
    )
    model.eval()

    result = benchmark_model(
        model=model,
        loader=loader,
        device=device,
        warmup_batches=benchmark_cfg.warmup_batches,
    )

    row = {
        "method": method_name,
        "model_config": method_cfg.model_config,
        "checkpoint_path": method_cfg.checkpoint_path,
        "split": benchmark_cfg.split,
        "limit": benchmark_cfg.limit,
        "batch_size": benchmark_cfg.batch_size,
        "warmup_batches": benchmark_cfg.warmup_batches,
        **result,
    }

    print("method: {}".format(method_name))
    print("total_images: {}".format(result["total_images"]))
    print("total_time_sec: {:.6f}".format(result["total_time_sec"]))
    print("ms_per_image: {:.6f}".format(result["ms_per_image"]))
    print("images_per_sec: {:.6f}".format(result["images_per_sec"]))

    del model

    if device == "cuda":
        torch.cuda.empty_cache()

    return row


def log_single_result(writer, row):
    if writer is None:
        return

    writer.set_step(0, "speed")

    method_name = row["method"]

    writer.add_scalar(
        "{}_ms_per_image".format(method_name),
        row["ms_per_image"],
    )
    writer.add_scalar(
        "{}_images_per_sec".format(method_name),
        row["images_per_sec"],
    )
    writer.add_scalar(
        "{}_total_time_sec".format(method_name),
        row["total_time_sec"],
    )

    table = pd.DataFrame([row])
    writer.add_table("reconstruction_speed_{}".format(method_name), table)


@hydra.main(
    version_base=None,
    config_path="../src/configs",
    config_name="benchmark_one_reconstruction_speed",
)
def main(cfg: DictConfig):
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = resolve_device(cfg.device)

    datasets = instantiate(cfg.datasets)
    dataset = get_dataset_by_split(datasets, cfg.benchmark.split)
    dataset = limit_dataset(dataset, cfg.benchmark.limit)

    loader = make_loader(
        dataset=dataset,
        batch_size=cfg.benchmark.batch_size,
        num_workers=cfg.benchmark.num_workers,
        pin_memory=False,
    )

    writer = build_writer(
        cfg=cfg,
        logger_name="benchmark_one_reconstruction_speed",
    )

    out_dir = Path(cfg.benchmark.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = run_single_speed_benchmark(
        method_cfg=cfg.method,
        loader=loader,
        benchmark_cfg=cfg.benchmark,
        device=device,
    )

    result_df = pd.DataFrame([row])
    out_path = out_dir / "reconstruction_speed_{}.csv".format(cfg.method.name)
    result_df.to_csv(out_path, index=False)

    print()
    print("Saved result to: {}".format(out_path))

    log_single_result(writer, row)


if __name__ == "__main__":
    main()
