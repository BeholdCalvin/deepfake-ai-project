"""
split_data.py

Splits raw .mp4 files into train / test sets.
Changes from original:
  • Accepts split ratio as an argument (default 80/20).
  • Prints a brief class-balance check after splitting.
  • Uses pathlib for cross-platform path handling.
  • Optional --seed argument for reproducible splits.
"""

import random
import shutil
import argparse
from pathlib import Path
from glob import glob


# ─────────────────────────────────────────────────────────────────────────────

def split_class(
    src_dir: str,
    train_dir: str,
    test_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Copies .mp4 files from `src_dir` into `train_dir` and `test_dir`.
    Returns (n_train, n_test).
    """
    files = glob(str(Path(src_dir) / "*.mp4"))

    if not files:
        print(f"  [skip] No .mp4 files found in {src_dir}")
        return 0, 0

    rng = random.Random(seed)
    rng.shuffle(files)

    split_idx   = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    test_files  = files[split_idx:]

    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy(f, train_dir)
    for f in test_files:
        shutil.copy(f, test_dir)

    print(
        f"  {Path(src_dir).name:>10}  →  "
        f"Train: {len(train_files):>4}  |  Test: {len(test_files):>4}"
    )
    return len(train_files), len(test_files)


# ─────────────────────────────────────────────────────────────────────────────

def main(split_ratio: float = 0.8, seed: int = 42):
    print(f"\nSplitting data  (ratio={split_ratio}, seed={seed})")
    print("-" * 50)

    r_tr, r_te = split_class(
        "data/raw/real",
        "data/raw/train/real",
        "data/raw/test/real",
        split_ratio=split_ratio,
        seed=seed,
    )
    f_tr, f_te = split_class(
        "data/raw/fake",
        "data/raw/train/fake",
        "data/raw/test/fake",
        split_ratio=split_ratio,
        seed=seed,
    )

    print("-" * 50)
    total_train = r_tr + f_tr
    total_test  = r_te + f_te

    if total_train > 0:
        fake_ratio_train = f_tr / total_train * 100
        print(f"  Train total: {total_train}  (fake: {fake_ratio_train:.1f} %)")

    if total_test > 0:
        fake_ratio_test = f_te / total_test * 100
        print(f"  Test  total: {total_test}   (fake: {fake_ratio_test:.1f} %)")

    print("Split complete.\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test split for raw video files.")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Fraction of files to use for training (default: 0.8)")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    main(split_ratio=args.ratio, seed=args.seed)
