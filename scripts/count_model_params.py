import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def count_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def load_config(config_name):
    config_dir = str(Path("src/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="train_unrolled_admm")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    overrides = []
    if args.model is not None:
        overrides.append(f"model={args.model}")

    config_dir = str(Path("src/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model = instantiate(cfg.model)

    total = count_params(model)

    print()
    print("Model parameters")
    print(f"model: {args.model or cfg.defaults}")
    print("total: {:.3f}M".format(total / 1e6))

    if hasattr(model, "pre"):
        print("pre: {:.3f}M".format(count_params(model.pre) / 1e6))

    if hasattr(model, "reconstructor"):
        print("reconstructor: {:.6f}M".format(count_params(model.reconstructor) / 1e6))

    if hasattr(model, "post"):
        print("post: {:.3f}M".format(count_params(model.post) / 1e6))


if __name__ == "__main__":
    main()
