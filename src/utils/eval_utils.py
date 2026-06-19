import logging

import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn


def limit_dataset(dataset, limit):
    """
    Limit dataset length by truncating its internal index.

    Args:
        dataset: Dataset object.
        limit: Maximum number of samples. If None or <= 0, no limit is used.

    Returns:
        Dataset object.
    """
    if limit is None:
        return dataset

    if limit <= 0:
        return dataset

    if hasattr(dataset, "_index"):
        dataset._index = dataset._index[:limit]

    return dataset


def make_loader(dataset, batch_size, num_workers, pin_memory=False):
    """
    Create dataloader for deterministic evaluation.

    Args:
        dataset: Dataset object.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.
        pin_memory: Whether to use pinned memory.

    Returns:
        DataLoader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_writer(cfg, logger_name):
    """
    Build Comet writer from Hydra config.

    Args:
        cfg: Hydra config.
        logger_name: Name for python logger.

    Returns:
        Writer object or None.
    """
    if not cfg.logging.log_comet:
        return None

    logger = logging.getLogger(logger_name)

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


def get_dataset_by_split(datasets, split):
    """
    Select dataset by split name.

    Args:
        datasets: Dict with train and val datasets.
        split: train, val, or test.

    Returns:
        Dataset object.
    """
    if split == "train":
        return datasets["train"]

    if split in ["val", "test"]:
        return datasets["val"]

    raise ValueError("Unknown split: {}".format(split))


def synchronize_if_needed(device):
    """
    Synchronize CUDA device before/after timing.

    Args:
        device: Device name.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_device(device):
    """
    Resolve device string.

    Args:
        device: Requested device.

    Returns:
        Available device.
    """
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return device
