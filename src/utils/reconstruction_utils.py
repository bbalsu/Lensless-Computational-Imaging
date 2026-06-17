from src.lensless_helpers.preprocessor import ALIGNMENT


def crop_roi_chw(x):
    """
    Crop tensor to the dataset ROI.

    Works with tensors whose last dimensions are [H, W], for example:
        [B, C, H, W]
        [C, H, W]
    """
    top, left = ALIGNMENT["top_left"]
    h = ALIGNMENT["height"]
    w = ALIGNMENT["width"]
    return x[..., top : top + h, left : left + w]


def tensor_to_image(x):
    """
    Convert CHW tensor in [0, 1] to HWC NumPy image for logging.
    """
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()
