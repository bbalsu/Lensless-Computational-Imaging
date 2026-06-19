from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from scripts.benchmark_one_reconstruction_speed import run_single_speed_benchmark
from src.utils.eval_utils import (
    build_writer,
    get_dataset_by_split,
    limit_dataset,
    make_loader,
    resolve_device,
)


def log_all_results(writer, results):
    if writer is None:
        return

    writer.set_step(0, "speed")

    table = pd.DataFrame(results)
    writer.add_table("reconstruction_speed_all_methods", table)

    for row in results:
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


@hydra.main(
    version_base=None,
    config_path="../src/configs",
    config_name="benchmark_reconstruction_speed",
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
        logger_name="benchmark_reconstruction_speed",
    )

    out_dir = Path(cfg.benchmark.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for method_cfg in cfg.methods:
        row = run_single_speed_benchmark(
            method_cfg=method_cfg,
            loader=loader,
            benchmark_cfg=cfg.benchmark,
            device=device,
        )
        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    out_path = out_dir / "reconstruction_speed_all_methods.csv"
    results_df.to_csv(out_path, index=False)

    print()
    print("Final speed benchmark results:")
    print(results_df)
    print()
    print("Saved results to: {}".format(out_path))

    log_all_results(writer, all_results)


if __name__ == "__main__":
    main()
