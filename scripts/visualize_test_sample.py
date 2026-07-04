import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.logger import setup_logging
from src.metrics.metric_utils import make_metric_objects
from src.utils.checkpoint_utils import load_model_checkpoint, resolve_checkpoint_path
from src.utils.reconstruction_utils import crop_roi_chw


def tensor_to_numpy_img(x):
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.permute(1, 2, 0).numpy()

def normalize_image_id(x):
    if pd.isna(x):
        raise ValueError("image_id is NaN")

    if isinstance(x, float) and x.is_integer():
        return str(int(x))

    text = str(x)
    if text.endswith(".0"):
        return text[:-2]

    return text

def get_requested_image_id(config):
    if config.get("image_id") is not None:
        return str(config.image_id)

    csv_path = config.get("csv_path")
    if csv_path is None:
        raise ValueError("Provide either +image_id=... or +csv_path=...")

    csv_path = Path(to_absolute_path(csv_path))
    df = pd.read_csv(csv_path)

    sort_by = config.get("sort_by", "psnr")
    index = int(config.get("index", 0))
    ascending = bool(config.get("ascending", False))

    if sort_by not in df.columns:
        raise ValueError(f"Column '{sort_by}' not found in {csv_path}")

    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    row = df.iloc[index]

    print("Selected row from CSV:")
    print(row)

    return normalize_image_id(row["image_id"])


@hydra.main(
    version_base=None,
    config_path="../src/configs",
    config_name="eval_reconstruction",
)
def main(config: DictConfig):
    output_dir = Path(to_absolute_path(config.get("output_dir", "outputs/visualizations")))
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    image_id = get_requested_image_id(config)
    logger.info(f"Requested image_id: {image_id}")

    datasets = instantiate(config.datasets)

    if config.split == "train":
        dataset = datasets["train"]
    elif config.split in ["test", "val"]:
        dataset = datasets["val"]
    else:
        raise ValueError(f"Unknown split: {config.split}")

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

    model.eval()

    metric_objects = make_metric_objects(config)

    sample_idx = int(image_id)
    sample = dataset[sample_idx]

    found_batch = collate_fn([sample])
    found_index = 0

    measurement = found_batch["measurement"].to(device)
    psf = found_batch["psf"].to(device)
    target_roi = found_batch["target_roi"].to(device)

    with torch.no_grad():
        reconstruction = model(measurement, psf)
        prediction_roi = crop_roi_chw(reconstruction)

    i = found_index

    pred = prediction_roi[i : i + 1].clamp(0, 1)
    target = target_roi[i : i + 1].clamp(0, 1)

    metrics = {}
    for metric in metric_objects:
        metrics[metric.name] = metric(prediction=pred, target=target)

    measurement_img = tensor_to_numpy_img(measurement[i])
    pred_img = tensor_to_numpy_img(pred[0])
    target_img = tensor_to_numpy_img(target[0])

    title_metrics = " | ".join(
        f"{name}: {value:.4f}" for name, value in metrics.items()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(measurement_img)
    axes[0].set_title("Lensless measurement")
    axes[0].axis("off")

    axes[1].imshow(pred_img)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(target_img)
    axes[2].set_title("Ground truth ROI")
    axes[2].axis("off")

    fig.suptitle(f"{model_name} | image_id={image_id}\n{title_metrics}", fontsize=12)
    plt.tight_layout()

    output_path = config.get("output_path")
    if output_path is None:
        output_path = output_dir / f"{model_name}_{config.split}_{image_id}.png"
    else:
        output_path = Path(to_absolute_path(output_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    logger.info(f"Saved visualization to: {output_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
