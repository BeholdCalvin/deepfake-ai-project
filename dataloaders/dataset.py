import os
import torch
from torch.utils.data import Dataset
import cv2
import glob

class DeepfakeSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5, transform=None, split=None):
        """
        data_dir structure:
        data_dir/
            real/
                video1_frame1.png, video1_frame2.png...
            fake/
                video2_frame1.png, video2_frame2.png...

        If `split` is provided and `data_dir` points to a parent directory,
        the dataset will try to resolve a matching subfolder automatically.
        """
        self.data_dir = self._resolve_data_dir(data_dir, split)
        self.sequence_length = sequence_length
        self.transform = transform
        self.split = split
        
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
        # Simplistic grouping by video name (assuming prefix before _frame is video ID)
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir): continue
            
            all_frames = sorted(glob.glob(os.path.join(class_dir, "*.png")))
            # Group into chunks of 'sequence_length'
            for i in range(0, len(all_frames) - self.sequence_length + 1, self.sequence_length):
                self.samples.append({
                    'frames': all_frames[i:i + self.sequence_length],
                    'label': label # 0 for real, 1 for fake
                })

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