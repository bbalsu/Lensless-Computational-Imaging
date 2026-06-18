import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
    get_roi,
)
from src.lensless_helpers.psf import simulate_psf_from_mask

logger = logging.getLogger(__name__)


class CustomDirDataset(BaseDataset):
    """
    Dataset for custom lensless data stored in a directory.

    Expected structure:
        data_dir/
        ├── lensless/
        │   ├── ImageID1.png
        │   └── ImageID2.png
        ├── masks/
        │   ├── ImageID1.npy
        │   └── ImageID2.npy
        └── lensed/  # optional
            ├── ImageID1.png
            └── ImageID2.png

    Each sample is returned in the same format as DigiCamHFDataset:
        measurement, target, target_roi, psf, mask_label, image_id.
    """

    def __init__(
        self,
        data_dir,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        self.data_dir = Path(data_dir)
        self.lensless_dir = self.data_dir / "lensless"
        self.masks_dir = self.data_dir / "masks"
        self.lensed_dir = self.data_dir / "lensed"

        self._check_dirs()

        self.has_targets = self.lensed_dir.exists()
        self._psf_cache = {}

        index = self._create_index()

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _check_dirs(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Custom data directory not found: {self.data_dir}")

        if not self.lensless_dir.exists():
            raise FileNotFoundError(f"Missing lensless directory: {self.lensless_dir}")

        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Missing masks directory: {self.masks_dir}")

    def _create_index(self):
        lensless_paths = sorted(self.lensless_dir.glob("*.png"))

        if len(lensless_paths) == 0:
            raise RuntimeError(f"No .png files found in {self.lensless_dir}")

        index = []

        for lensless_path in lensless_paths:
            image_id = lensless_path.stem
            mask_path = self.masks_dir / f"{image_id}.npy"
            target_path = self.lensed_dir / f"{image_id}.png"

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask file for image_id={image_id} was not found: {mask_path}"
                )

            if self.has_targets and not target_path.exists():
                logger.warning(
                    "Target file for image_id=%s was not found: %s",
                    image_id,
                    target_path,
                )
                target_path = None

            index.append(
                {
                    "lensless_path": str(lensless_path),
                    "mask_path": str(mask_path),
                    "target_path": str(target_path)
                    if target_path is not None
                    else None,
                    "label": -1,
                    "image_id": image_id,
                }
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "lensless_path" in entry, "Each item must include 'lensless_path'."
            assert "mask_path" in entry, "Each item must include 'mask_path'."
            assert "target_path" in entry, "Each item must include 'target_path'."
            assert "label" in entry, "Each item must include 'label'."
            assert "image_id" in entry, "Each item must include 'image_id'."

    @staticmethod
    def _load_image(path):
        image = Image.open(path).convert("RGB")
        image = convert_image_to_float(force_rgb(np.array(image)))
        return image

    @staticmethod
    def _to_chw(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]

        return x.permute(2, 0, 1).contiguous().float()

    def _get_psf(self, mask_path):
        if mask_path in self._psf_cache:
            return self._psf_cache[mask_path]

        mask = np.load(mask_path)
        psf = simulate_psf_from_mask(mask)
        psf = self._to_chw(psf)

        self._psf_cache[mask_path] = psf
        return psf

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        lensless_hwc = self._load_image(data_dict["lensless_path"])

        psf = self._get_psf(data_dict["mask_path"])

        instance_data = {
            "measurement": self._to_chw(lensless_hwc),
            "psf": psf.clone(),
            "mask_label": torch.tensor(data_dict["label"], dtype=torch.long),
            "image_id": data_dict["image_id"],
        }

        if data_dict["target_path"] is not None:
            target_hwc = self._load_image(data_dict["target_path"])
            target_roi = get_roi(target_hwc)

            instance_data["target"] = self._to_chw(torch.from_numpy(target_hwc))
            instance_data["target_roi"] = self._to_chw(target_roi)

        instance_data = self.preprocess_data(instance_data)
        return instance_data
