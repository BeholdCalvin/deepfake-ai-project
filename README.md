# 🔍 Multi-Domain Deepfake Detector

A deep learning system for detecting AI-generated face manipulations in images and videos, built with a dual-stream architecture that analyses both spatial and frequency-domain artefacts simultaneously.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://deepfake-ai-project-beholdcalvin.streamlit.app/)

---

## 🧠 How It Works

Most deepfake detectors rely on a single stream of evidence. This project fuses two complementary perspectives:

- **Spatial Branch (EfficientNet-B4)** — detects pixel-level artefacts: soft jaw edges, skin texture mismatches, and blending boundaries that GAN face-swaps leave behind.
- **Frequency Branch (FFT-CNN)** — detects spectral fingerprints: the periodic grid patterns and ring artefacts introduced by GAN up-sampling, which survive JPEG re-encoding and social-media compression.

Frame-level features from both branches are fused and passed through a **bidirectional-ready LSTM**, allowing the model to reason across temporal sequences in video rather than evaluating each frame independently.

---

## ✨ Features

- Upload **video** (MP4, AVI, MOV) or **still image** (JPG, PNG) for instant analysis
- **Dual-Branch Grad-CAM** visualisation — highlights exactly which face regions and spectral patterns triggered the verdict, for both branches independently
- Colour-coded **verdict badge** (FAKE / REAL) with a calibrated confidence score
- Sidebar controls for frame sampling rate and face crop alignment mode
- Robust to heavy compression — training augmentations simulate social-media re-encoding (JPEG/WebP, downscaling, motion blur)
- Identity-aware data splitting ensures the model learns manipulation artefacts rather than memorising faces

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | **98.75%** |
| ROC-AUC | **0.9997** |
| F1 Score | **0.9874** |

![ROC Curve](image.png)

---

## 🏗️ Architecture

```
Input (image or video sequence)
        │
        ├──► EfficientNet-B4 (spatial) ──► 1792-d features
        │
        └──► FFT-CNN (frequency)       ──►  512-d features
                                               │
                                          Concat (2304-d)
                                               │
                                          Linear + LayerNorm
                                               │
                                          LSTM (temporal, video only)
                                               │
                                          Classifier → logit → sigmoid
```

---

## 🗂️ Project Structure

```
├── app.py              # Streamlit frontend
├── train.py            # Training loop (Focal Loss, CosineAnnealingLR)
├── evaluate.py         # Evaluation (Accuracy, AUC, F1, ECE, confusion matrix)
├── predict.py          # Inference + Dual-Branch Grad-CAM
├── preprocess.py       # Offline face extraction from raw videos (MTCNN)
├── split_data.py       # Identity-aware train/test split (zero subject leakage)
├── models/
│   ├── fusion.py       # DeepfakeDetector — dual-stream model + checkpoint utils
│   └── branches.py     # Pixel / Frequency / Spatial branch modules
├── dataloaders/
│   ├── dataset.py      # Video-coherent sequence dataset
│   └── transforms.py   # Compression-robust augmentation pipeline
└── utils/
    ├── face_extractor.py  # MTCNN + landmark-aligned face crops
    ├── metrics.py         # FocalLoss, accuracy, AUC
    └── visualizer.py      # FFT spectrum plotting
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/BeholdCalvin/deepfake-ai-project.git
cd deepfake-ai-project
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Model weights (`weights/best_model.pth`) are not included in the repo due to file size.


---

## 🛠️ Tech Stack

`PyTorch` · `EfficientNet-B4 (timm)` · `MTCNN (facenet-pytorch)` · `Albumentations` · `Streamlit` · `scikit-learn` · `OpenCV`

