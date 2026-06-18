from pathlib import Path

import torch

from src.utils.io_utils import ROOT_PATH

CHECKPOINTS = {
    "unrolled_admm": {
        "file_id": "1cEgfugM5LhkfajfSuLcPQUqAwUhS4j6v",
        "path": "checkpoints/unrolled_admm/best.pth",
    },
    "leadmm5_pre4_post4": {
        "file_id": "1Li3VQCsB-P9YbHjR-pT-ekW2ec11DCa0",
        "path": "checkpoints/leadmm5_pre4_post4/best.pth",
    },
    "leadmm5_pre8": {
        "file_id": "1MSa-abFSt4d1VKyxGTT_uBdTkadtKXkt",
        "path": "checkpoints/leadmm5_pre8/best.pth",
    },
    "leadmm5_post8": {
        "file_id": "1tyMii4Z-x79lkASPdOVGkGvXJE0JqaU0",
        "path": "checkpoints/leadmm5_post8/best.pth",
    },
}


def get_default_checkpoint_path(model_name):
    if model_name == "admm":
        return None

    if model_name not in CHECKPOINTS:
        raise ValueError(
            f"No default checkpoint is registered for model='{model_name}'. "
            f"Available models: {list(CHECKPOINTS.keys()) + ['admm']}"
        )

    return ROOT_PATH / CHECKPOINTS[model_name]["path"]


def download_checkpoint(model_name, force=False):
    if model_name == "admm":
        print("model=admm does not require a checkpoint.")
        return None

    if model_name not in CHECKPOINTS:
        raise ValueError(
            f"No checkpoint URL is registered for model='{model_name}'. "
            f"Available models: {list(CHECKPOINTS.keys())}"
        )

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Checkpoint auto-download requires gdown. "
            "Install it with: pip install gdown"
        ) from exc

    checkpoint_info = CHECKPOINTS[model_name]
    checkpoint_path = ROOT_PATH / checkpoint_info["path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and not force:
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path

    url = f"https: //drive.google.com/uc?id={checkpoint_info['file_id']}"
    print(f"Downloading checkpoint for {model_name} to {checkpoint_path}")
    gdown.download(url, str(checkpoint_path), quiet=False)

    if not checkpoint_path.exists():
        raise RuntimeError(f"Failed to download checkpoint to: {checkpoint_path}")

    return checkpoint_path


def resolve_checkpoint_path(model_name, from_pretrained=None, auto_download=True):
    if model_name == "admm" and from_pretrained is None:
        return None

    if from_pretrained is not None:
        checkpoint_path = Path(from_pretrained)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT_PATH / checkpoint_path
    else:
        checkpoint_path = get_default_checkpoint_path(model_name)

    if checkpoint_path is None:
        return None

    if checkpoint_path.exists():
        return checkpoint_path

    if auto_download and from_pretrained is None:
        return download_checkpoint(model_name)

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. "
        "Either provide inferencer.from_pretrained=... or enable "
        "inferencer.auto_download=true."
    )


def load_model_checkpoint(model, checkpoint_path, device, strict=True):
    if checkpoint_path is None:
        return model

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        fixed_state_dict[key] = value

    model.load_state_dict(fixed_state_dict, strict=strict)
    return model
