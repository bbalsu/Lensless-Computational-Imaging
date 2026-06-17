import csv
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.datasets.collate import collate_fn
from src.lensless_helpers.preprocessor import ALIGNMENT
from src.logger import setup_logging
from src.metrics.tracker import MetricTracker
from src.metrics.utils import make_metric_objects
from src.utils.reconstruction_utils import crop_roi_chw, tensor_to_image


@hydra.main(version_base=None, config_path="../src/configs", config_name="eval_admm")
def main(config: DictConfig):
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    logger = logging.getLogger(__name__)

    logger.info("Resolved config:")
    logger.info(OmegaConf.to_yaml(config))

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA is not available, switching to CPU.")
        device = "cpu"

    writer = None
    if config.logging.log_comet:
        project_config = OmegaConf.to_container(config, resolve=True)
        writer = instantiate(
            config.writer,
            logger=logger,
            project_config=project_config,
            _recursive_=False,
        )

    datasets = instantiate(config.datasets)

    if config.split == "train":
        dataset = datasets["train"]
    elif config.split in ["test", "val"]:
        dataset = datasets["val"]
    else:
        raise ValueError(f"Unknown split: {config.split}")

    if config.limit is not None:
        dataset._index = dataset._index[: config.limit]

    loader = instantiate(
        config.dataloader,
        dataset=dataset,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = instantiate(config.model).to(device)
    model.eval()

    metric_objects = make_metric_objects(config)
    metric_names = [metric.name for metric in metric_objects]
    metric_tracker = MetricTracker(*metric_names, writer=writer)

    rows = []

    logger.info(f"split: {config.split}")
    logger.info(f"samples: {len(dataset)}")
    logger.info(f"num_iters: {config.model.num_iters}")
    logger.info(f"psf_gain: {float(config.model.psf_gain)}")
    logger.info(f"metrics: {metric_names}")

    logged_images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            measurement = batch["measurement"].to(device)
            target_roi = batch["target_roi"].to(device)
            psf = batch["psf"].to(device)

            mask_labels = batch["mask_label"].detach().cpu().tolist()
            image_ids = batch["image_id"]

            reconstruction = model(measurement, psf)
            reconstruction_roi = crop_roi_chw(reconstruction)
            reconstruction_roi_clipped = reconstruction_roi.clamp(0, 1)

            for i in range(reconstruction_roi.shape[0]):
                pred = reconstruction_roi_clipped[i : i + 1]
                target = target_roi[i : i + 1]

                row = {
                    "image_id": image_ids[i],
                    "mask_label": int(mask_labels[i]),
                    "psf_gain": float(config.model.psf_gain),
                    "rec_min": reconstruction_roi[i].min().item(),
                    "rec_max": reconstruction_roi[i].max().item(),
                    "rec_mean": reconstruction_roi[i].mean().item(),
                }

                for metric in metric_objects:
                    value = metric(prediction=pred, target=target)
                    row[metric.name] = value
                    metric_tracker.update(metric.name, value, n=1)

                rows.append(row)

                if (
                    writer is not None
                    and logged_images < config.logging.log_first_n_images
                ):
                    writer.set_step(logged_images, mode=config.split)
                    writer.add_image(
                        f"measurement_{image_ids[i]}",
                        tensor_to_image(measurement[i]),
                    )
                    writer.add_image(
                        f"target_roi_{image_ids[i]}",
                        tensor_to_image(target_roi[i]),
                    )
                    writer.add_image(
                        f"admm_reconstruction_roi_{image_ids[i]}",
                        tensor_to_image(pred[0]),
                    )
                    logged_images += 1

    final_metrics = metric_tracker.result()

    logger.info("")
    logger.info("Final metrics")
    for name, value in final_metrics.items():
        logger.info("{}: {:.6f}".format(name, value))

    result_path = out_dir / f"admm_{config.split}_results.csv"
    summary_path = out_dir / f"admm_{config.split}_summary.txt"

    with open(result_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    with open(summary_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
        f.write("\n")
        f.write(f"samples: {len(dataset)}\n")
        for name, value in final_metrics.items():
            f.write("{}: {:.6f}\n".format(name, value))

    logger.info(f"Saved per-sample results to: {result_path}")
    logger.info(f"Saved summary to: {summary_path}")

    if writer is not None:
        writer.set_step(0, mode=config.split)
        writer.add_scalars(
            {
                **{f"mean_{name}": value for name, value in final_metrics.items()},
                "num_samples": len(dataset),
                "psf_gain": float(config.model.psf_gain),
            }
        )

        if config.logging.log_table:
            df = pd.DataFrame(rows)
            writer.add_table(f"admm_{config.split}_results", df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
