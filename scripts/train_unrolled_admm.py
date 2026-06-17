import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.datasets.collate import collate_fn
from src.logger import setup_logging
from src.metrics.metric_utils import make_metric_objects
from src.utils.io_utils import ROOT_PATH


def apply_limit(dataset, limit):
    """
    Restrict dataset length for debug runs.
    """
    if limit is not None and limit > 0:
        dataset._index = dataset._index[:limit]
    return dataset


def make_loader(config, dataset, shuffle):
    """
    Instantiate DataLoader from Hydra config.
    """
    return instantiate(
        config.dataloader,
        dataset=dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


@hydra.main(
    version_base=None,
    config_path="../src/configs",
    config_name="train_unrolled_admm",
)
def main(config: DictConfig):
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(save_dir)
    logger = logging.getLogger(__name__)

    logger.info("Resolved config:")
    logger.info(OmegaConf.to_yaml(config))

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA is not available, switching to CPU.")
        device = "cpu"

    datasets = instantiate(config.datasets)

    train_dataset = apply_limit(datasets["train"], config.train_limit)
    val_dataset = apply_limit(datasets["val"], config.val_limit)

    dataloaders = {
        "train": make_loader(config, train_dataset, shuffle=True),
        "val": make_loader(config, val_dataset, shuffle=False),
    }

    model = instantiate(config.model).to(device)
    criterion = instantiate(config.loss).to(device)
    optimizer = instantiate(config.optimizer, params=model.parameters())

    lr_scheduler = None
    if config.get("lr_scheduler") is not None:
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    metric_objects = make_metric_objects(config)
    metrics = {
        "train": metric_objects,
        "inference": metric_objects,
    }

    writer = None
    if config.logging.log_comet:
        project_config = OmegaConf.to_container(config, resolve=True)
        writer = instantiate(
            config.writer,
            logger=logger,
            project_config=project_config,
            _recursive_=False,
        )

    trainer = instantiate(
        config.trainer,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms={},
        _recursive_=False,
    )

    trainer.train()


if __name__ == "__main__":
    main()
