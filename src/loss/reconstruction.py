import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_clip01(x):
    return x.clamp(0.0, 1.0)


class ReconstructionLoss(nn.Module):
    """
    loss = mse_weight * MSE(prediction, target)
          + lpips_weight * LPIPS(prediction, target)

    LPIPS parameters are frozen, but gradients still flow through LPIPS
    features to the reconstruction tensor.
    """

    def __init__(
        self,
        mse_weight=1.0,
        lpips_weight=0.1,
        lpips_net="vgg",
        clip=True,
        device="auto",
    ):
        """
        Args:
            mse_weight: Weight of pixel-wise MSE loss.
            lpips_weight: Weight of perceptual LPIPS loss.
            lpips_net: LPIPS backbone - "vgg"
            clip: Whether to clamp inputs to [0, 1] before loss calculation.
            device: Device for LPIPS network.
        """
        super().__init__()

        self.mse_weight = float(mse_weight)
        self.lpips_weight = float(lpips_weight)
        self.clip = clip

        self.lpips_model = None
        if self.lpips_weight > 0:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "ReconstructionLoss with LPIPS requires lpips. "
                    "Install it with: pip install lpips"
                ) from exc

            self.lpips_model = lpips.LPIPS(net=lpips_net).to(device)
            self.lpips_model.eval()

            for param in self.lpips_model.parameters():
                param.requires_grad = False

    @staticmethod
    def _to_lpips_range(x):
        """
        Convert image range from [0, 1] to [-1, 1].
        """
        return x * 2.0 - 1.0

    def forward(self, prediction, target):
        """
        Args:
            prediction: Reconstructed ROI tensor [B, C, H, W].
            target: Ground-truth ROI tensor [B, C, H, W].

        Returns:
            Dictionary with total loss and detached scalar components.
        """
        if self.clip:
            prediction = _safe_clip01(prediction)
            target = _safe_clip01(target)

        mse_loss = F.mse_loss(prediction, target)
        total_loss = self.mse_weight * mse_loss

        lpips_loss = prediction.new_tensor(0.0)
        if self.lpips_model is not None:
            pred_lpips = self._to_lpips_range(prediction)
            target_lpips = self._to_lpips_range(target)
            lpips_loss = self.lpips_model(pred_lpips, target_lpips).mean()
            total_loss = total_loss + self.lpips_weight * lpips_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss.detach(),
            "lpips_loss": lpips_loss.detach(),
        }
