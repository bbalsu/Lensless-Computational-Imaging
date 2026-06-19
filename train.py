import os
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "scripts" / "train_unrolled_admm.py"

    os.execv(
        sys.executable,
        [sys.executable, str(script_path), *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
