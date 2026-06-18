import argparse

from src.utils.checkpoint_utils import CHECKPOINTS, download_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *CHECKPOINTS.keys()],
        help="Model checkpoint to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload checkpoint even if it already exists.",
    )
    args = parser.parse_args()

    if args.model == "all":
        for model_name in CHECKPOINTS:
            download_checkpoint(model_name, force=args.force)
    else:
        download_checkpoint(args.model, force=args.force)


if __name__ == "__main__":
    main()
