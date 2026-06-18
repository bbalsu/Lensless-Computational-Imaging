import argparse
import shutil
import zipfile
from pathlib import Path


def normalize_extracted_dir(out_dir):
    """
    Ensure that out_dir directly contains lensless/ and masks/.

    If zip contains a single root folder, move its contents to out_dir.
    """
    if (out_dir / "lensless").exists() and (out_dir / "masks").exists():
        return out_dir

    children = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        nested = children[0]
        if (nested / "lensless").exists() and (nested / "masks").exists():
            tmp_dir = out_dir.parent / f"{out_dir.name}_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

            nested.rename(tmp_dir)
            shutil.rmtree(out_dir)
            tmp_dir.rename(out_dir)
            return out_dir

    raise RuntimeError(
        "Extracted dataset must contain lensless/ and masks/ directories. "
        f"Got contents: {[p.name for p in out_dir.iterdir()]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/custom_demo")
    parser.add_argument("--zip-path", type=str, default="data/custom_demo.zip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zip_path = Path(args.zip_path)

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)

    if (
        out_dir.exists()
        and (out_dir / "lensless").exists()
        and (out_dir / "masks").exists()
    ):
        print(f"Dataset already exists: {out_dir}")
        print(f"DATA_DIR={out_dir.resolve()}")
        return

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install gdown with: pip install gdown") from exc

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset zip to: {zip_path}")
    gdown.download(args.url, str(zip_path), quiet=False, fuzzy=True)

    if not zip_path.exists():
        raise RuntimeError(f"Download failed: {zip_path}")

    print(f"Extracting to: {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    data_dir = normalize_extracted_dir(out_dir)

    print()
    print("Dataset is ready.")
    print(f"DATA_DIR={data_dir.resolve()}")


if __name__ == "__main__":
    main()
