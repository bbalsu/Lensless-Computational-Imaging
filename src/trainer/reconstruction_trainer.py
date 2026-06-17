from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.reconstruction_utils import crop_roi_chw, tensor_to_image


class ReconstructionTrainer(BaseTrainer):
    """
    Trainer for lensless reconstruction models.
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
        **kwargs,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad(set_to_none=True)

        reconstruction = self.model(batch["measurement"], batch["psf"])
        prediction_roi = crop_roi_chw(reconstruction)
        target_roi = batch["target_roi"]

        loss_dict = self.criterion(prediction_roi, target_roi)

        batch["reconstruction"] = reconstruction
        batch["prediction_roi"] = prediction_roi
        batch["target_roi"] = target_roi
        batch.update(loss_dict)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        batch_size = prediction_roi.shape[0]

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item(), n=batch_size)

        prediction_roi = prediction_roi.clamp(0, 1)
        target_roi = target_roi.clamp(0, 1)

        for metric in metric_funcs:
            value = metric(prediction=prediction_roi, target=target_roi)
            metrics.update(metric.name, value, n=batch_size)

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if self.writer is None:
            return

        log_first_n_images = self.config.trainer.get("log_first_n_images", 0)
        if log_first_n_images <= 0:
            return

        if mode == "train" and batch_idx != 0:
            return

        batch_size = batch["prediction_roi"].shape[0]
        n_images = min(batch_size, log_first_n_images)

        for i in range(n_images):
            self.writer.add_image(
                "{}_measurement_{}_{}".format(mode, batch_idx, i),
                tensor_to_image(batch["measurement"][i]),
            )
            self.writer.add_image(
                "{}_target_roi_{}_{}".format(mode, batch_idx, i),
                tensor_to_image(batch["target_roi"][i]),
            )
            self.writer.add_image(
                "{}_prediction_roi_{}_{}".format(mode, batch_idx, i),
                tensor_to_image(batch["prediction_roi"][i]),
            )
