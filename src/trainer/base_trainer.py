from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
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
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms or {}

        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(f"    {key: 15s}: {value}")

            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        Args:
            epoch (int): current training epoch.

        Returns:
            logs (dict): logs with average loss and metrics.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()

        if self.writer is not None:
            self.writer.set_step((epoch - 1) * self.epoch_len)
            self.writer.add_scalar("epoch", epoch)

        last_train_metrics = {}

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx % self.log_step == 0:
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)

                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )

                if self.writer is not None:
                    if self.lr_scheduler is not None:
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("learning rate", lr)

                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on.
            dataloader (DataLoader): dataloader for the partition.

        Returns:
            logs (dict): evaluation logs.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(batch, metrics=self.evaluation_metrics)

            if self.writer is not None:
                self.writer.set_step(epoch * self.epoch_len, part)

            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics.
        """
        best = False
        stop_process = False

        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True

        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch.

        Returns:
            batch (dict): batch with tensors on device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Apply batch transforms.

        Args:
            batch (dict): dict-based batch.

        Returns:
            batch (dict): transformed batch.
        """
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)

        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )

        return batch

    def _clip_grad_norm(self):
        """
        Clip gradient norm by config.trainer.max_grad_norm.
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculate gradient norm.

        Args:
            norm_type (float | str | None): norm order.

        Returns:
            total_norm (float): calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculate percentage of processed batch within epoch.

        Args:
            batch_idx (int): current batch index.

        Returns:
            progress (str): current step and percentage.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len

        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer class.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after process_batch.
            mode (str): train or inference.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return

        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, save as model_best.pth.
            only_best (bool): if True, do not duplicate best checkpoint.
        """
        arch = type(self.model).__name__

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")

        if not (only_best and save_best):
            torch.save(state, filename)
            if self.writer is not None and self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.writer is not None and self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint.

        Args:
            resume_path (str): path to checkpoint.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(
            resume_path,
            map_location=self.device,
            weights_only=False,
        )

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file "
                "is different from that of the checkpoint. This may yield an "
                "exception when state_dict is loaded."
            )

        self.model.load_state_dict(checkpoint["state_dict"])

        if checkpoint["config"]["optimizer"] != self.config["optimizer"] or checkpoint[
            "config"
        ].get("lr_scheduler") != self.config.get("lr_scheduler"):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is "
                "different from that of the checkpoint. Optimizer and scheduler "
                "parameters are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if (
                self.lr_scheduler is not None
                and checkpoint.get("lr_scheduler") is not None
            ):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Args:
            pretrained_path (str): path to model state dict.
        """
        pretrained_path = str(pretrained_path)

        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")

        checkpoint = torch.load(
            pretrained_path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif checkpoint.get("model_state_dict") is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
