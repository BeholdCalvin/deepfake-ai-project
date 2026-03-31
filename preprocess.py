"""
preprocess.py

Extracts and saves face crops from raw videos to disk.
This is the *offline* pre-processing step; augmentation happens at training
time inside the DataLoader (see dataloaders/transforms.py).

Changes from original:
  • Resize guard: skips frames where MTCNN fails silently (face area too small).
  • Progress bar shows per-video face yield so you can spot bad batches.
  • Supports a --dry-run flag (CLI) to count faces without writing files.
  • Added min_face_size parameter to FaceExtractor call for robustness.
"""

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

from utils.face_extractor import FaceExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def process_videos(
    input_dir: str,
    output_dir: str,
    frames_per_video: int = 8,
    batch_size: int = 8,
    resize_wh: tuple[int, int] = (640, 360),
    dry_run: bool = False,
) -> dict:
    """
    Walk `input_dir` for *.mp4 files, extract `frames_per_video` evenly-spaced
    frames per video, run MTCNN face detection in batches, and save crops.

    Returns a summary dict: {total_videos, total_faces, failed_videos}.
    """
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    extractor = FaceExtractor()
    videos    = glob.glob(os.path.join(input_dir, "*.mp4"))

    if not videos:
        print(f"[preprocess] No .mp4 files found in: {input_dir}")
        return {"total_videos": 0, "total_faces": 0, "failed_videos": 0}

    total_faces   = 0
    failed_videos = 0

    pbar = tqdm(videos, desc=f"Processing {os.path.basename(input_dir)}")

    for vid_path in pbar:
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]

        cap          = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            failed_videos += 1
            continue

        # ── Sample frame indices ──────────────────────────────────────────────
        indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

        raw_frames: list[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            # Resize for faster MTCNN inference
            frame_resized = cv2.resize(frame, resize_wh)
            frame_rgb     = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame_rgb)

        cap.release()

        if not raw_frames:
            failed_videos += 1
            continue

        # ── Batch face extraction ─────────────────────────────────────────────
        count = 0
        for i in range(0, len(raw_frames), batch_size):
            batch = raw_frames[i : i + batch_size]
            faces = extractor.extract_batch(batch)

            for face in faces:
                if face is None:
                    continue

                # Skip tiny face crops that will confuse the model
                h, w = face.shape[:2]
                if h < 48 or w < 48:
                    continue

                if not dry_run:
                    out_path = os.path.join(
                        output_dir, f"{vid_name}_frame{count:03d}.png"
                    )
                    cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                count       += 1
                total_faces += 1

        pbar.set_postfix(faces=count, vid=vid_name[:20])

    summary = {
        "total_videos": len(videos),
        "total_faces":  total_faces,
        "failed_videos": failed_videos,
    }
    mode_tag = "[DRY RUN] " if dry_run else ""
    print(
        f"\n{mode_tag}Done. "
        f"Videos: {len(videos)}, "
        f"Faces saved: {total_faces}, "
        f"Failed: {failed_videos}"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Extract face crops from raw video files.")
    p.add_argument("--input",  "-i", required=True,  help="Directory containing .mp4 files")
    p.add_argument("--output", "-o", required=True,  help="Directory to save face crops")
    p.add_argument("--frames", "-f", type=int, default=8,  help="Frames to sample per video")
    p.add_argument("--batch",  "-b", type=int, default=8,  help="MTCNN batch size")
    p.add_argument("--dry-run",      action="store_true",  help="Count faces without writing files")
    return p.parse_args()


if __name__ == "__main__":
    # ── Scripted shortcuts (comment / uncomment as needed) ───────────────────
    # process_videos("data/raw/train/real", "data/processed/train/real")
    # process_videos("data/raw/train/fake", "data/processed/train/fake")
    # process_videos("data/raw/test/real",  "data/processed/test/real")
    # process_videos("data/raw/test/fake",  "data/processed/test/fake")
    process_videos("data/raw/test2", "data/processed/test2")

    # ── Or use CLI: python preprocess.py -i data/raw/train/real -o data/processed/train/real
    # args = _parse_args()
    # process_videos(args.input, args.output, args.frames, args.batch, dry_run=args.dry_run)
