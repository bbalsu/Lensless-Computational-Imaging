import sys
import types
import urllib.request
from pathlib import Path
import contextlib
import io

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tvf


def _patch_torchvision_functional_tensor():
    module_name = "torchvision.transforms.functional_tensor"

    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.rgb_to_grayscale = tvf.rgb_to_grayscale
    sys.modules[module_name] = module


class RealESRGANRestorer:
    """
    Frozen Real-ESRGAN post-processor for ADMM reconstructions.

    The model is a general-purpose GAN-based image restoration network.
    It is applied after fixed ADMM-100 and is not trained on lensless data.
    """

    WEIGHTS_URL = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    )

    def __init__(
        self,
        weights_path="checkpoints/restoration/RealESRGAN_x4plus.pth",
        device="cuda",
        tile=256,
        half=False,
    ):
        """
        Args:
            weights_path: Path to Real-ESRGAN weights.
            device: Device for inference.
            tile: Tile size for memory-efficient inference.
            half: Whether to use FP16 on CUDA.
        """
        _patch_torchvision_functional_tensor()

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.weights_path = Path(weights_path)
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.weights_path.exists():
            print("Downloading Real-ESRGAN weights to:", self.weights_path)
            urllib.request.urlretrieve(
                self.WEIGHTS_URL,
                str(self.weights_path),
            )

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        gpu_id = 0 if device == "cuda" else None

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.weights_path),
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half and device == "cuda",
            gpu_id=gpu_id,
        )

    @staticmethod
    def _tensor_to_bgr_uint8(image):
        image = image.detach().cpu().clamp(0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255.0).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def _bgr_uint8_to_tensor(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    def restore_tensor(self, image, out_hw=None):
        """
        Restore one image.

        Args:
            image: RGB tensor [3, H, W] in [0, 1].
            out_hw: Optional output size (H, W).

        Returns:
            Restored RGB tensor [3, H, W] in [0, 1].
        """
        h, w = image.shape[-2:]

        if out_hw is None:
            out_hw = (h, w)

        bgr = self._tensor_to_bgr_uint8(image)

        with contextlib.redirect_stdout(io.StringIO()):
            restored, _ = self.upsampler.enhance(
                bgr,
                outscale=1,
            )

        if restored.shape[:2] != tuple(out_hw):
            restored = cv2.resize(
                restored,
                (out_hw[1], out_hw[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        restored = self._bgr_uint8_to_tensor(restored)
        return restored

    def __call__(self, images):
        """
        Restore a batch of images.

        Args:
            images: RGB tensor [B, 3, H, W] in [0, 1].

        Returns:
            Restored RGB tensor [B, 3, H, W] in [0, 1].
        """
        out_hw = images.shape[-2:]
        restored = []

        for image in images:
            restored.append(self.restore_tensor(image, out_hw=out_hw))

        return torch.stack(restored, dim=0).to(images.device)
