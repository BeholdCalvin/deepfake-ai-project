import torch
from facenet_pytorch import MTCNN
from PIL import Image


class FaceExtractor:
    def __init__(self, device='cuda', margin=0.3):
        print(f"[INFO] Using device: {device}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            keep_all=False,
            select_largest=True,
            device=self.device
        )
        self.margin = margin

    def extract_batch(self, frames_rgb):
        imgs = [Image.fromarray(f) for f in frames_rgb]

        boxes, _ = self.mtcnn.detect(imgs)

        results = []

        for frame, box in zip(frames_rgb, boxes):
            if box is None:
                results.append(None)
                continue

            box = box[0]
            x1, y1, x2, y2 = [int(b) for b in box]

            h, w, _ = frame.shape

            bw, bh = x2 - x1, y2 - y1

            x1 = max(0, int(x1 - bw * self.margin))
            y1 = max(0, int(y1 - bh * self.margin))
            x2 = min(w, int(x2 + bw * self.margin))
            y2 = min(h, int(y2 + bh * self.margin))

            results.append(frame[y1:y2, x1:x2])

        return results