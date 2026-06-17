from pathlib import Path

import requests
from tqdm.auto import tqdm


def main():
    out_dir = Path("data/masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K/resolve/main/masks"

    for label in tqdm(range(100)):
        out_path = out_dir / f"mask_{label}.npy"

        if out_path.exists():
            continue

        url = f"{base_url}/mask_{label}.npy"
        response = requests.get(url)
        response.raise_for_status()

        out_path.write_bytes(response.content)


if __name__ == "__main__":
    main()
