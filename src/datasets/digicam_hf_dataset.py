import logging
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
    get_roi,
)
from src.lensless_helpers.psf import simulate_psf_from_mask

logger = logging.getLogger(__name__)


class DigiCamHFDataset(BaseDataset):
    """
    Dataset wrapper for DigiCam-Mirflickr-MultiMask-10K.

    Each sample contains a lensless measurement, a lensed target image,
    the ROI target, the corresponding simulated PSF, the mask label,
    and an image id.
    """

    def __init__(
        self,
        split="train",
        masks_dir="data/masks",
        hf_repo="bezzam/DigiCam-Mirflickr-MultiMask-10K",
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Load the HuggingFace split and build a lightweight index.

        Args:
            split: Dataset split name, usually "train" or "test".
            masks_dir: Directory with mask_{label}.npy files.
            hf_repo: HuggingFace dataset repository name.
            limit: Optional maximum number of samples.
            shuffle_index: Whether to shuffle sample index.
            instance_transforms: Optional transforms applied to one sample.

        Output:
            Initialized dataset with lazy image loading and cached PSFs.
        """
        self.split = split
        self.masks_dir = Path(masks_dir)
        self.hf_repo = hf_repo

        self.dataset = load_dataset(hf_repo, split=split)
        self._psf_cache = {}

        index = self._create_index()

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(self):
        """
        Build metadata-only index for lazy image loading.

        Input:
            HuggingFace dataset stored in self.dataset.

        Returns:
            List of dictionaries with HuggingFace row index, mask label,
            and image id.
        """
        index = []
        mask_labels = self.dataset["mask_label"]
        for i, mask_label in enumerate(mask_labels):
            index.append(
                {
                    "hf_idx": i,
                    "label": int(mask_label),
                    "image_id": f"{i: 06d}",
                }
            )
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Validate index entries used by this HuggingFace-backed dataset.

        Args:
            index: List of dataset index entries.

        Output:
            Raises AssertionError if a required key is missing.
        """
        for entry in index:
            assert "hf_idx" in entry, "Each item must include 'hf_idx'."
            assert "label" in entry, "Each item must include 'label' (mask_label)."
            assert "image_id" in entry, "Each item must include 'image_id'."

    def _load_mask(self, mask_label):
        """
        Load a raw mask pattern from disk and cache it.

        Args:
            mask_label: Integer mask label from the dataset.

        Returns:
            NumPy array with raw LCD mask pattern.
        """
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if mask_label in self._mask_cache:
            return self._mask_cache[mask_label]

        mask_path = self.masks_dir / f"mask_{mask_label}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {mask_path}. "
                f"Run scripts/download_masks.py first."
            )

        mask = np.load(mask_path)
        self._mask_cache[mask_label] = mask
        return mask

    def _get_psf(self, mask_label):
        """
        Simulate and cache PSF for a given mask label.

        Args:
            mask_label: Integer mask label from the dataset.

        Returns:
            PSF tensor with shape [C, H, W].
        """
        if mask_label in self._psf_cache:
            return self._psf_cache[mask_label]

        mask = self._load_mask(mask_label)
        psf = simulate_psf_from_mask(mask)
        psf = self._to_chw(psf)

        self._psf_cache[mask_label] = psf
        return psf

    @staticmethod
    def _format_images(lensed, lensless):
        """
        Convert PIL images to float tensors and align them with helper logic.

        Args:
            lensed: Ground truth PIL image from the dataset.
            lensless: Lensless measurement PIL image from the dataset.

        Returns:
            Tuple of tensors (lensed_hwc, lensless_hwc), both in HWC layout
            and float range [0, 1].
        """
        lensed = convert_image_to_float(force_rgb(np.array(lensed)))
        lensless = convert_image_to_float(force_rgb(np.array(lensless)))

        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)

        lensed = get_cropped_lensed(lensed, lensless)
        lensed = torch.from_numpy(lensed)

        return lensed, lensless

    @staticmethod
    def _to_chw(x):
        """
        Convert HWC or 1HWC tensor to CHW float tensor.

        Args:
            x: Tensor with shape [H, W, C] or [1, H, W, C].

        Returns:
            Float tensor with shape [C, H, W].
        """
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        return x.permute(2, 0, 1).contiguous().float()

    def __getitem__(self, ind):
        """
        Return one formatted dataset sample.

        Args:
            ind: Dataset index.

        Returns:
            Dictionary with:
                measurement: Lensless image tensor [C, H, W].
                target: Full aligned lensed image tensor [C, H, W].
                target_roi: ROI target tensor [C, H_roi, W_roi].
                psf: Simulated PSF tensor [C, H, W].
                mask_label: Mask label tensor.
                image_id: String image id.
        """
        data_dict = self._index[ind]
        item = self.dataset[data_dict["hf_idx"]]
        mask_label = int(item["mask_label"])

        lensed_hwc, lensless_hwc = self._format_images(item["lensed"], item["lensless"])
        psf = self._get_psf(mask_label)
        target_roi = get_roi(lensed_hwc)

        instance_data = {
            "measurement": self._to_chw(lensless_hwc),
            "target": self._to_chw(lensed_hwc),
            "target_roi": self._to_chw(target_roi),
            "psf": psf.clone(),
            "mask_label": torch.tensor(mask_label, dtype=torch.long),
            "image_id": data_dict["image_id"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data
