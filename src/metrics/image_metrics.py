import torch

from src.metrics.base_metric import BaseMetric


def _safe_clip01(x):
    return x.clamp(0.0, 1.0)


class MSEMetric(BaseMetric):
    def __init__(self, clip=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **batch):
        if self.clip:
            prediction = _safe_clip01(prediction)
            target = _safe_clip01(target)
        return torch.mean((prediction - target) ** 2).item()


class PSNRMetric(BaseMetric):
    def __init__(self, data_range=1.0, clip=True, eps=1e-12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_range = float(data_range)
        self.clip = clip
        self.eps = eps

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **batch):
        if self.clip:
            prediction = _safe_clip01(prediction)
            target = _safe_clip01(target)

        mse = torch.mean((prediction - target) ** 2).clamp_min(self.eps)
        data_range = torch.tensor(
            self.data_range, device=prediction.device, dtype=prediction.dtype
        )
        psnr = 20 * torch.log10(data_range) - 10 * torch.log10(mse)
        return psnr.item()


class SSIMMetric(BaseMetric):
    def __init__(self, data_range=1.0, clip=True, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_range = float(data_range)
        self.clip = clip

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
        except ImportError as exc:
            raise ImportError(
                "SSIMMetric requires torchmetrics. Install it with: pip install torchmetrics"
            ) from exc

        self.metric = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(
            device
        )

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **batch):
        if self.clip:
            prediction = _safe_clip01(prediction)
            target = _safe_clip01(target)

        value = self.metric(prediction, target)
        return value.item()


class LPIPSMetric(BaseMetric):
    def __init__(self, net="vgg", clip=True, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPSMetric requires lpips. Install it with: pip install lpips"
            ) from exc

        self.metric = lpips.LPIPS(net=net).to(device)
        self.metric.eval()

    @staticmethod
    def _to_lpips_range(x):
        return x * 2.0 - 1.0

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **batch):
        if self.clip:
            prediction = _safe_clip01(prediction)
            target = _safe_clip01(target)

        prediction = self._to_lpips_range(prediction)
        target = self._to_lpips_range(target)

        with torch.no_grad():
            value = self.metric(prediction, target)

        return value.mean().item()
