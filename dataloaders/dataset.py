import os
import re
import random
import torch
from torch.utils.data import Dataset
import cv2
import glob
from collections import defaultdict

# preprocess.py writes face crops as "{video_stem}_frame{count:03d}.png".
# This regex strips the "_frameNNN" suffix to recover the video ID that a
# given frame belongs to, e.g. "000_003_frame005.png" -> "000_003".
_FRAME_SUFFIX_RE = re.compile(r"_frame\d+$")


def _video_id_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _FRAME_SUFFIX_RE.search(stem)
    return stem[: match.start()] if match else stem


class DeepfakeSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5, transform=None, split=None, seed=42):
        """
        data_dir structure:
        data_dir/
            real/
                000_frame000.png, 000_frame001.png, ..., 001_frame000.png, ...
            fake/
                000_003_frame000.png, 000_003_frame001.png, ...

        Frames are grouped by VIDEO ID (the filename prefix before
        "_frameNNN") before being chunked into sequences. This is critical:
        a naive sort-and-chunk over the flat file list interleaves frames
        from different videos into the same "sequence" whenever a video's
        frame count isn't a clean multiple of sequence_length, which feeds
        the LSTM temporally-incoherent, spliced-together inputs and badly
        degrades both training signal and reported accuracy.

        If `split` is provided and `data_dir` points to a parent directory,
        the dataset will try to resolve a matching subfolder automatically.
        """
        self.data_dir = self._resolve_data_dir(data_dir, split)
        self.sequence_length = sequence_length
        self.transform = transform
        self.split = split
        self._rng = random.Random(seed)

        self.samples = []
        self._prepare_data()

    @staticmethod
    def _resolve_data_dir(data_dir, split=None):
        if not split:
            return data_dir

        normalized_split = split.lower()
        candidate_names = [normalized_split]
        if normalized_split == "val":
            candidate_names.append("test")

        for candidate_name in candidate_names:
            candidate_dir = os.path.join(data_dir, candidate_name)
            if os.path.isdir(candidate_dir):
                return candidate_dir

        return data_dir

    def _prepare_data(self):
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            all_frames = sorted(glob.glob(os.path.join(class_dir, "*.png")))

            # Group frames by video ID so every sequence comes from ONE video.
            frames_by_video = defaultdict(list)
            for frame_path in all_frames:
                frames_by_video[_video_id_from_filename(frame_path)].append(frame_path)

            for video_id, frames in frames_by_video.items():
                frames = sorted(frames)  # ensure temporal order within the video

                if len(frames) >= self.sequence_length:
                    # Non-overlapping chunks of sequence_length, all from
                    # this single video. Any short remainder at the end
                    # (< sequence_length frames) is dropped rather than
                    # mixed with the next video.
                    for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length):
                        self.samples.append({
                            'frames': frames[i:i + self.sequence_length],
                            'label': label,
                            'video_id': video_id,
                        })
                elif len(frames) > 0:
                    # Short video (fewer extracted faces than sequence_length,
                    # e.g. due to occlusion/detection misses): pad by
                    # repeating the last frame instead of silently dropping
                    # the video's data entirely.
                    padded = frames + [frames[-1]] * (self.sequence_length - len(frames))
                    self.samples.append({
                        'frames': padded,
                        'label': label,
                        'video_id': video_id,
                    })

        # Shuffle once at construction time. Samples are still grouped
        # per-video internally (each sample's frames all share one video_id),
        # but the order of samples in self.samples no longer follows the
        # real-videos-then-fake-videos glob order, which matters if anything
        # downstream (e.g. a quick sanity-check slice) assumes interleaving.
        self._rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_tensors = []
        
        for frame_path in sample['frames']:
            # Read image using cv2, convert BGR to RGB
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=img)
                img_tensor = augmented['image']
            else:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                
            frame_tensors.append(img_tensor)
            
        # Stack into [T, C, H, W]
        sequence_tensor = torch.stack(frame_tensors)
        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
        return sequence_tensor, label_tensor