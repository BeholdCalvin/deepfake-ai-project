"""
split_data.py

Zero-leakage train/test split: videos are partitioned at the subject-identity
level so that no person's face appears in both splits.

The problem with naive file-level random splits
────────────────────────────────────────────────
FaceForensics++ (and most deepfake datasets) contain many fake videos per real
identity.  A random 80/20 file split can place '000.mp4' (real, identity 000)
in train and '000_003.mp4' (fake, same identity) in test.  The model then
learns to recognise *person-level* facial features rather than *manipulation*
artefacts, inflating test metrics without improving generalisation.

The solution implemented here
─────────────────────────────
1. Extract a primary subject ID from every filename (first numeric token by
   default, matching FF++ naming: '000_003.mp4' → '000').
2. Build the identity universe from BOTH real AND fake filenames so the split
   is globally consistent: a given identity always lands in the same partition
   across both classes.
3. Split at the subject level with three layers of mathematical assertions:
     a) train_subjects ∩ test_subjects = ∅          (no subject in both splits)
     b) |train| + |test| = |all_subjects|           (no subject lost)
     c) Cross-class: real_train_ids ∩ fake_test_ids = ∅  (and vice versa)
4. Emit a warning when any fake video involves two subjects from different
   partitions (a known limitation of primary-ID-only splitting).

Filename convention assumed (FaceForensics++ / common datasets)
───────────────────────────────────────────────────────────────
  Real : '000.mp4', '001.mp4', ...
  Fake : '000_003.mp4', '001_000.mp4', ...
  Primary identity key = first numeric token in the stem.

Override with --id-pattern for datasets with different naming conventions.
"""

import re
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_subject_id(filename: str, id_pattern: str = r"^(\d+)") -> str:
    """
    Derive the primary subject / identity token from a filename.

    Examples with the default pattern r'^(\\d+)':
      '000.mp4'         → '000'   (real video, identity 000)
      '000_003.mp4'     → '000'   (fake: 000 is the primary/source identity)
      'subject042.mp4'  → '042'
      'unknown_clip.mp4'→ 'unknown_clip'  (no digits → stem used as fallback)

    Args:
        filename:   Bare filename string (not a full path).
        id_pattern: Regex with exactly one capture group.

    Returns:
        Non-empty subject ID string.
    """
    stem  = Path(filename).stem
    match = re.search(id_pattern, stem)
    return match.group(1) if match else stem


def _glob_mp4(directory: str) -> list[Path]:
    """Return a sorted list of .mp4 paths in *directory* (non-recursive)."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob("*.mp4"))


def _ids_in_dir(directory: str, id_pattern: str) -> set[str]:
    """Return the set of subject IDs derived from all .mp4 files in a directory."""
    return {
        _extract_subject_id(p.name, id_pattern)
        for p in _glob_mp4(directory)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core split logic
# ─────────────────────────────────────────────────────────────────────────────

def split_by_identity(
    real_src:       str,
    fake_src:       str,
    train_real_dir: str,
    test_real_dir:  str,
    train_fake_dir: str,
    test_fake_dir:  str,
    split_ratio:    float = 0.8,
    seed:           int   = 42,
    id_pattern:     str   = r"^(\d+)",
) -> dict[str, int]:
    """
    Identity-aware train/test split for a two-class (real / fake) dataset.

    Partitions at the *subject* level so that every video (real AND fake)
    belonging to a given identity lands in the same split, preventing the
    model from learning identity-specific appearance instead of manipulation
    artefacts.

    Args:
        real_src:       Source directory of real .mp4 files.
        fake_src:       Source directory of fake .mp4 files.
        train_real_dir: Destination for training real files.
        test_real_dir:  Destination for test real files.
        train_fake_dir: Destination for training fake files.
        test_fake_dir:  Destination for test fake files.
        split_ratio:    Fraction of subject IDs assigned to training (default 0.8).
        seed:           RNG seed for reproducibility (default 42).
        id_pattern:     Regex (one capture group) to extract subject ID from stem.

    Returns:
        Dict with counts: n_train_real, n_test_real, n_train_fake, n_test_fake.

    Raises:
        AssertionError: if any zero-leakage guarantee is violated (should never
                        happen for valid inputs; acts as a correctness proof).
    """
    real_files = _glob_mp4(real_src)
    fake_files = _glob_mp4(fake_src)

    if not real_files and not fake_files:
        print(f"  [skip] No .mp4 files found in '{real_src}' or '{fake_src}'")
        return dict(n_train_real=0, n_test_real=0, n_train_fake=0, n_test_fake=0)

    # ── Group files by primary subject ID ─────────────────────────────────
    real_by_subject: dict[str, list[Path]] = defaultdict(list)
    fake_by_subject: dict[str, list[Path]] = defaultdict(list)

    for f in real_files:
        real_by_subject[_extract_subject_id(f.name, id_pattern)].append(f)
    for f in fake_files:
        fake_by_subject[_extract_subject_id(f.name, id_pattern)].append(f)

    # Union across both classes: a subject that appears only in fake videos
    # (e.g. a target identity with no corresponding real clip in this subset)
    # is still handled correctly by the partition logic.
    all_subjects: list[str] = sorted(
        real_by_subject.keys() | fake_by_subject.keys()
    )

    if not all_subjects:
        return dict(n_train_real=0, n_test_real=0, n_train_fake=0, n_test_fake=0)

    # ── Partition subjects ─────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_subjects)

    split_idx      = max(1, int(len(all_subjects) * split_ratio))
    train_subjects = set(all_subjects[:split_idx])
    test_subjects  = set(all_subjects[split_idx:])

    # ── Assertion layer 1: subject-level mutual exclusivity ───────────────
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, (
        f"PARTITION ERROR – {len(overlap)} subject(s) in both splits: "
        f"{sorted(overlap)[:10]}"
    )
    assert len(train_subjects) + len(test_subjects) == len(all_subjects), (
        "PARTITION ERROR – subjects were lost during splitting. "
        f"Expected {len(all_subjects)}, got "
        f"{len(train_subjects) + len(test_subjects)}."
    )
    print(
        f"  ✓ Subject-level assertion passed: "
        f"{len(train_subjects)} train / {len(test_subjects)} test subjects, "
        f"overlap = 0"
    )

    # ── Create output directories ─────────────────────────────────────────
    for d in (train_real_dir, test_real_dir, train_fake_dir, test_fake_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Copy files into their assigned partition ───────────────────────────
    def _copy_class(
        by_subject: dict[str, list[Path]],
        train_dir: str,
        test_dir:  str,
    ) -> tuple[int, int]:
        n_train = n_test = 0
        for sid, files in by_subject.items():
            dst = train_dir if sid in train_subjects else test_dir
            for f in files:
                shutil.copy(f, dst)
                if dst == train_dir:
                    n_train += 1
                else:
                    n_test  += 1
        return n_train, n_test

    n_tr_r, n_te_r = _copy_class(real_by_subject, train_real_dir, test_real_dir)
    n_tr_f, n_te_f = _copy_class(fake_by_subject, train_fake_dir, test_fake_dir)

    # ── Assertion layer 2: file-level cross-class verification ───────────
    # These hold by construction (both classes use the same partition map)
    # but explicit assertions serve as executable documentation and catch
    # any future refactoring that breaks the invariant.
    real_train_ids = _ids_in_dir(train_real_dir, id_pattern)
    real_test_ids  = _ids_in_dir(test_real_dir,  id_pattern)
    fake_train_ids = _ids_in_dir(train_fake_dir, id_pattern)
    fake_test_ids  = _ids_in_dir(test_fake_dir,  id_pattern)

    # Within-class: no subject in both train and test for the same class
    assert len(real_train_ids & real_test_ids) == 0, \
        f"REAL-CLASS LEAKAGE: {real_train_ids & real_test_ids}"
    assert len(fake_train_ids & fake_test_ids) == 0, \
        f"FAKE-CLASS LEAKAGE: {fake_train_ids & fake_test_ids}"

    # Cross-class: a subject in real-train must not appear in fake-test, and v.v.
    # This guarantees the model cannot use identity recognition as a shortcut.
    cross_rt_ft = real_train_ids & fake_test_ids
    cross_re_fr = real_test_ids  & fake_train_ids
    assert len(cross_rt_ft) == 0, \
        f"CROSS-CLASS LEAKAGE (real-train ∩ fake-test): {cross_rt_ft}"
    assert len(cross_re_fr) == 0, \
        f"CROSS-CLASS LEAKAGE (real-test ∩ fake-train): {cross_re_fr}"

    print("  ✓ Cross-class leakage assertions passed.")

    # ── Assertion layer 3: mixed-partition fake-video warning ─────────────
    # A fake video '000_003.mp4' involves TWO identities (000 and 003).
    # Primary-ID splitting guarantees 000 is consistent, but 003 might be in
    # the opposite partition.  This is a known limitation of filename-based
    # splitting without full face-recognition clustering.  The warning below
    # quantifies the exposure so the researcher can decide whether it matters.
    mixed_count = 0
    all_numeric_ids = re.compile(r"(\d+)")
    for f in fake_files:
        ids_in_name = all_numeric_ids.findall(Path(f.name).stem)
        if len(ids_in_name) >= 2:
            primary_in_train   = ids_in_name[0] in train_subjects
            secondary_in_train = ids_in_name[1] in train_subjects
            if primary_in_train != secondary_in_train:
                mixed_count += 1

    if mixed_count > 0:
        print(
            f"  ⚠  {mixed_count} fake video(s) involve subjects from different "
            f"partitions (secondary-ID partial leakage). "
            f"For full isolation use the dataset's official split JSON or "
            f"face-recognition-based identity clustering."
        )
    else:
        print("  ✓ No mixed-partition fake videos detected.")

    return dict(
        n_train_real=n_tr_r,
        n_test_real=n_te_r,
        n_train_fake=n_tr_f,
        n_test_fake=n_te_f,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    split_ratio: float = 0.8,
    seed:        int   = 42,
    id_pattern:  str   = r"^(\d+)",
) -> None:
    print(
        f"\nSplitting data  "
        f"(ratio={split_ratio}, seed={seed}, id_pattern={id_pattern!r})"
    )
    print("-" * 60)

    counts = split_by_identity(
        real_src       = "data/raw/real",
        fake_src       = "data/raw/fake",
        train_real_dir = "data/raw/train/real",
        test_real_dir  = "data/raw/test/real",
        train_fake_dir = "data/raw/train/fake",
        test_fake_dir  = "data/raw/test/fake",
        split_ratio    = split_ratio,
        seed           = seed,
        id_pattern     = id_pattern,
    )

    print("-" * 60)
    total_train = counts["n_train_real"] + counts["n_train_fake"]
    total_test  = counts["n_test_real"]  + counts["n_test_fake"]

    if total_train > 0:
        fake_pct = counts["n_train_fake"] / total_train * 100
        print(f"  Train: {total_train:>5} files  "
              f"(real: {counts['n_train_real']}, fake: {counts['n_train_fake']}, "
              f"fake%: {fake_pct:.1f})")
    if total_test > 0:
        fake_pct = counts["n_test_fake"] / total_test * 100
        print(f"  Test:  {total_test:>5} files  "
              f"(real: {counts['n_test_real']}, fake: {counts['n_test_fake']}, "
              f"fake%: {fake_pct:.1f})")

    print("Split complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identity-aware train/test split for deepfake video datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8,
        help="Fraction of subject IDs to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    parser.add_argument(
        "--id-pattern", type=str, default=r"^(\d+)",
        dest="id_pattern",
        help=(
            r"Regex (one capture group) to extract subject ID from filename stem. "
            r"Default r'^(\d+)' works for FF++ naming ('000_003' → '000')."
        ),
    )
    args = parser.parse_args()
    main(split_ratio=args.ratio, seed=args.seed, id_pattern=args.id_pattern)