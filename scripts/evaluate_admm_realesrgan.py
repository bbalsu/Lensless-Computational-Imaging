import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.datasets.collate import collate_fn
from src.metrics.metric_utils import make_metric_objects
from src.metrics.tracker import MetricTracker
from src.model.fft_operators import center_crop


def crop_like(image, target):
    if image.shape[-2:] == target.shape[-2:]:
        return image
    return center_crop(image, target.shape[-2:])


def limit_dataset(dataset, limit):
    if limit is None:
        return dataset

    if limit <= 0:
        return dataset

    if hasattr(dataset, "_index"):
        dataset._index = dataset._index[:limit]

    return dataset


def make_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_writer(cfg):
    if not cfg.logging.log_comet:
        return None

    logger = logging.getLogger("evaluate_admm_realesrgan")

    try:
        writer = instantiate(
            cfg.writer,
            logger=logger,
            project_config=cfg,
        )
        return writer
    except Exception as exc:
        print("Comet writer was not created:", exc)
        return None


def save_and_log_images(
    writer,
    measurement,
    target,
    admm_pred,
    restored_pred,
    out_dir,
    split_name,
    global_index,
    max_images,
):
    measurement = crop_like(measurement, target)

    batch_size = measurement.shape[0]
    saved_count = 0

    for i in range(batch_size):
        if global_index + i >= max_images:
            break

        images = torch.stack(
            [
                measurement[i].detach().cpu().clamp(0.0, 1.0),
                admm_pred[i].detach().cpu().clamp(0.0, 1.0),
                restored_pred[i].detach().cpu().clamp(0.0, 1.0),
                target[i].detach().cpu().clamp(0.0, 1.0),
            ],
            dim=0,
        )

        grid = make_grid(images, nrow=4)

        filename = "{}_sample_{:04d}.png".format(
            split_name,
            global_index + i,
        )
        out_path = out_dir / filename
        save_image(grid, out_path)

        if writer is not None:
            writer.set_step(global_index + i, split_name)
            writer.add_image(
                "comparison_{:04d}".format(global_index + i),
                str(out_path),
            )

        saved_count += 1

    return saved_count


def update_metrics(tracker, metrics, prefix, prediction, target, batch_size):
    for metric in metrics:
        value = metric(prediction=prediction, target=target)
        tracker.update(
            "{}_{}".format(prefix, metric.name),
            value,
            n=batch_size,
        )


def evaluate_split(
    cfg,
    split_name,
    dataset,
    admm_model,
    restorer,
    metrics,
    writer,
    device,
    out_dir,
):
    dataset = limit_dataset(dataset, cfg.evaluation.limit)

    loader = make_loader(
        dataset=dataset,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )

    metric_keys = []
    for metric in metrics:
        metric_keys.append("admm_{}".format(metric.name))
        metric_keys.append("restored_{}".format(metric.name))

    tracker = MetricTracker(*metric_keys)

    logged_images = 0

    for batch in tqdm(loader, desc="Evaluating {}".format(split_name)):
        measurement = batch["measurement"].to(device)
        psf = batch["psf"].to(device)
        target = batch["target_roi"].to(device)

        with torch.no_grad():
            admm_pred = admm_model(measurement, psf)
            admm_pred = crop_like(admm_pred, target)
            admm_pred = admm_pred.clamp(0.0, 1.0)

        restored_pred = restorer(admm_pred)
        restored_pred = restored_pred.clamp(0.0, 1.0)

        batch_size = target.shape[0]

        with torch.no_grad():
            update_metrics(
                tracker=tracker,
                metrics=metrics,
                prefix="admm",
                prediction=admm_pred,
                target=target,
                batch_size=batch_size,
            )

            update_metrics(
                tracker=tracker,
                metrics=metrics,
                prefix="restored",
                prediction=restored_pred,
                target=target,
                batch_size=batch_size,
            )

        if logged_images < cfg.evaluation.log_first_n_images:
            remaining = cfg.evaluation.log_first_n_images - logged_images
            logged_now = save_and_log_images(
                writer=writer,
                measurement=measurement,
                target=target,
                admm_pred=admm_pred,
                restored_pred=restored_pred,
                out_dir=out_dir,
                split_name=split_name,
                global_index=logged_images,
                max_images=logged_images + remaining,
            )
            logged_images += logged_now

    results = tracker.result()

    print()
    print("Results for split: {}".format(split_name))
    for key, value in results.items():
        print("{}: {:.8f}".format(key, value))

    if writer is not None:
        writer.set_step(0, split_name)
        writer.add_scalars(results)

        rows = []
        for method_prefix, method_name in [
            ("admm", "ADMM-100"),
            ("restored", "ADMM-100 + Real-ESRGAN"),
        ]:
            row = {"method": method_name}

            for metric in metrics:
                key = "{}_{}".format(method_prefix, metric.name)
                row[metric.name] = results[key]

            rows.append(row)

        table = pd.DataFrame(rows)
        writer.add_table("{}_metrics".format(split_name), table)

    return results


@hydra.main(
    version_base=None,
    config_path="../src/configs",
    config_name="eval_admm_realesrgan",
)
def main(cfg: DictConfig):
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    datasets = instantiate(cfg.datasets)

    admm_model = instantiate(cfg.model).to(device)
    admm_model.eval()

    restorer = instantiate(cfg.restoration)

    metrics = make_metric_objects(cfg)

    writer = build_writer(cfg)

    out_dir = Path(cfg.evaluation.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.evaluation.split == "train":
        evaluate_split(
            cfg=cfg,
            split_name="train",
            dataset=datasets["train"],
            admm_model=admm_model,
            restorer=restorer,
            metrics=metrics,
            writer=writer,
            device=device,
            out_dir=out_dir,
        )

    elif cfg.evaluation.split in ["val", "test"]:
        evaluate_split(
            cfg=cfg,
            split_name="val",
            dataset=datasets["val"],
            admm_model=admm_model,
            restorer=restorer,
            metrics=metrics,
            writer=writer,
            device=device,
            out_dir=out_dir,
        )

    elif cfg.evaluation.split == "both":
        evaluate_split(
            cfg=cfg,
            split_name="train",
            dataset=datasets["train"],
            admm_model=admm_model,
            restorer=restorer,
            metrics=metrics,
            writer=writer,
            device=device,
            out_dir=out_dir,
        )

        evaluate_split(
            cfg=cfg,
            split_name="val",
            dataset=datasets["val"],
            admm_model=admm_model,
            restorer=restorer,
            metrics=metrics,
            writer=writer,
            device=device,
            out_dir=out_dir,
        )

    else:
        raise ValueError("Unknown split: {}".format(cfg.evaluation.split))


if __name__ == "__main__":
    main()
