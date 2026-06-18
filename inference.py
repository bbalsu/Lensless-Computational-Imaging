import warnings
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.checkpoint_utils import load_model_checkpoint, resolve_checkpoint_path
from src.utils.init_utils import set_random_seed
from src.utils.reconstruction_utils import crop_roi_chw

warnings.filterwarnings("ignore", category=UserWarning)


def move_batch_to_device(batch, device, device_tensors):
    for key in device_tensors:
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device)
    return batch


def save_image_tensor(tensor, path):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)

    if tensor.ndim == 4:
        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C, H, W], got {tuple(tensor.shape)}")

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype("uint8")

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_dataloader(config):
    datasets = instantiate(config.datasets)

    if isinstance(datasets, (dict, DictConfig)):
        if config.inferencer.dataset_part is None:
            if len(datasets) != 1:
                raise ValueError(
                    "inferencer.dataset_part is null, but config.datasets "
                    f"contains several parts: {list(datasets.keys())}. "
                    "Please specify inferencer.dataset_part=..."
                )
            dataset_part = list(datasets.keys())[0]
        else:
            dataset_part = config.inferencer.dataset_part

        if dataset_part not in datasets:
            raise KeyError(
                f"Dataset part '{dataset_part}' was not found. "
                f"Available parts: {list(datasets.keys())}"
            )

        dataset = datasets[dataset_part]
    else:
        dataset = datasets

    return DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device
    device = torch.device(device)

    model_name = HydraConfig.get().runtime.choices["model"]

    dataloader = build_dataloader(config)

    model = instantiate(config.model).to(device)

    checkpoint_path = resolve_checkpoint_path(
        model_name=model_name,
        from_pretrained=config.inferencer.from_pretrained,
        auto_download=config.inferencer.auto_download,
    )

    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            strict=config.inferencer.strict_load,
        )
    else:
        print("No checkpoint is used.")

    model.eval()

    output_dir = Path(to_absolute_path(config.inferencer.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            image_ids = batch["image_id"]

            batch = move_batch_to_device(
                batch=batch,
                device=device,
                device_tensors=config.inferencer.device_tensors,
            )

            reconstruction = model(batch["measurement"], batch["psf"])
            reconstruction = crop_roi_chw(reconstruction)

            for i, image_id in enumerate(image_ids):
                save_path = output_dir / f"{image_id}.png"
                save_image_tensor(reconstruction[i], save_path)
                saved += 1

    print(f"Saved {saved} reconstructions to: {output_dir}")


if __name__ == "__main__":
    main()
