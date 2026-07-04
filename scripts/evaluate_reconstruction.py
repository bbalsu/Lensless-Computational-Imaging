import csv
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.logger import setup_logging
from src.metrics.metric_utils import make_metric_objects
from src.metrics.tracker import MetricTracker
from src.utils.checkpoint_utils import load_model_checkpoint, resolve_checkpoint_path
from src.utils.reconstruction_utils import crop_roi_chw, tensor_to_image
from src.utils.eval_utils import limit_dataset


@hydra.main(
    version_base=None,
    config_path="../src/configs",
    config_name="eval_reconstruction",
)
def main(config: DictConfig):
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    logger = logging.getLogger(__name__)

    logger.info("Resolved config:")
    logger.info(OmegaConf.to_yaml(config))

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA is not available, switching to CPU.")
        device = "cpu"
    device = torch.device(device)

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
        # In src/configs/datasets/digicam_hf.yaml, datasets["val"]
        # is the official DigiCam HF split="test".
        dataset = datasets["val"]
    else:
        raise ValueError(f"Unknown split: {config.split}")

    dataset = limit_dataset(dataset, config.limit)

    loader = instantiate(
        config.dataloader,
        dataset=dataset,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model_name = HydraConfig.get().runtime.choices["model"]
    model = instantiate(config.model).to(device)

    checkpoint_path = resolve_checkpoint_path(
        model_name=model_name,
        from_pretrained=config.evaluator.from_pretrained,
        auto_download=config.evaluator.auto_download,
    )

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model = load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            strict=config.evaluator.strict_load,
        )
    else:
        logger.info("No checkpoint is used.")

    model.eval()

    metric_objects = make_metric_objects(config)
    metric_names = [metric.name for metric in metric_objects]
    metric_tracker = MetricTracker(*metric_names, writer=writer)

    rows = []
    logged_images = 0

    logger.info(f"model: {model_name}")
    logger.info(f"split: {config.split}")
    logger.info(f"samples: {len(dataset)}")
    logger.info(f"checkpoint: {checkpoint_path}")
    logger.info(f"metrics: {metric_names}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluate {config.split}")):
            measurement = batch["measurement"].to(device)
            psf = batch["psf"].to(device)
            target_roi = batch["target_roi"].to(device)

            image_ids = batch["image_id"]
            mask_labels = batch.get("mask_label", None)
            if mask_labels is not None and torch.is_tensor(mask_labels):
                mask_labels = mask_labels.detach().cpu().tolist()

            reconstruction = model(measurement, psf)
            prediction_roi = crop_roi_chw(reconstruction)
            prediction_roi_clipped = prediction_roi.clamp(0, 1)
            target_roi_clipped = target_roi.clamp(0, 1)

            for i in range(prediction_roi.shape[0]):
                pred = prediction_roi_clipped[i : i + 1]
                target = target_roi_clipped[i : i + 1]

                row = {
                    "image_id": image_ids[i],
                    "rec_min": prediction_roi[i].min().item(),
                    "rec_max": prediction_roi[i].max().item(),
                    "rec_mean": prediction_roi[i].mean().item(),
                }
                if mask_labels is not None:
                    row["mask_label"] = int(mask_labels[i])

                for metric in metric_objects:
                    value = metric(prediction=pred, target=target)
                    row[metric.name] = value
                    metric_tracker.update(metric.name, value, n=1)

                rows.append(row)

                if writer is not None and logged_images < config.logging.log_first_n_images:
                    writer.set_step(logged_images, mode=config.split)
                    writer.add_image(
                        f"measurement_{image_ids[i]}",
                        tensor_to_image(measurement[i]),
                    )
                    writer.add_image(
                        f"target_roi_{image_ids[i]}",
                        tensor_to_image(target_roi_clipped[i]),
                    )
                    writer.add_image(
                        f"prediction_roi_{image_ids[i]}",
                        tensor_to_image(pred[0]),
                    )
                    logged_images += 1

    final_metrics = metric_tracker.result()

    logger.info("")
    logger.info("Final metrics")
    for name, value in final_metrics.items():
        logger.info(f"{name}: {value:.6f}")

    result_path = out_dir / f"{model_name}_{config.split}_results.csv"
    summary_path = out_dir / f"{model_name}_{config.split}_summary.txt"

    if rows:
        with open(result_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(rows)
    else:
        logger.warning("No rows were evaluated; CSV was not saved.")

    with open(summary_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
        f.write("\n")
        f.write(f"model: {model_name}\n")
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"samples: {len(dataset)}\n")
        for name, value in final_metrics.items():
            f.write(f"{name}: {value:.6f}\n")

    logger.info(f"Saved per-sample results to: {result_path}")
    logger.info(f"Saved summary to: {summary_path}")

    if writer is not None:
        writer.set_step(0, mode=config.split)
        writer.add_scalars(
            {
                **{f"mean_{name}": value for name, value in final_metrics.items()},
                "num_samples": len(dataset),
            }
        )

        if config.logging.log_table and rows:
            df = pd.DataFrame(rows)
            writer.add_table(f"{model_name}_{config.split}_results", df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
