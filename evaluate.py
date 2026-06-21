"""
evaluate.py

Runs a full evaluation pass on a held-out dataset and reports:
  • Accuracy
  • ROC-AUC
  • F1 Score
  • ECE  (Expected Calibration Error – how well confidence mirrors accuracy)
  • Confusion matrix counts

Compatible with the updated DeepfakeDetector (EfficientNet-B4 backbone).
No AMP needed during eval; torch.no_grad() is sufficient.
"""

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataloaders.dataset import DeepfakeSequenceDataset
from dataloaders.transforms import get_val_transforms
from models.fusion import DeepfakeDetector, load_checkpoint
from utils.metrics import calculate_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metric
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error.
    Lower is better; 0.0 = perfectly calibrated.

    We use 15 bins (vs the original 10) for a finer resolution with larger
    test sets.  The formula is the confidence-weighted absolute difference
    between average confidence and average accuracy in each bin.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (probs > low) & (probs <= high)
        prop   = in_bin.mean()
        if prop > 0:
            acc_bin  = labels[in_bin].mean()
            conf_bin = probs[in_bin].mean()
            ece     += np.abs(conf_bin - acc_bin) * prop

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open("configs/eval_dfdc.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    val_transform = get_val_transforms(config["image_size"])
    val_dataset   = DeepfakeSequenceDataset(
        data_dir=config["data_dir"],
        sequence_length=config["sequence_length"],
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepfakeDetector(
        sequence_length=config["sequence_length"],
        hidden_size=config.get("hidden_size", 512),
        lstm_layers=config.get("lstm_layers", 2),
    ).to(device)

    # load_checkpoint() tolerates the known legacy Grad-CAM alias keys from a
    # transitional version of fusion.py (see models/fusion.py) while still
    # raising loudly (strict=True) on any genuine architecture mismatch.
    # It also calls model.eval() internally.
    load_checkpoint(model, config["weights_path"], device)

    # ── Inference loop ────────────────────────────────────────────────────────
    all_labels: list[float] = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Evaluating"):
            sequences = sequences.to(device, non_blocking=True)
            logits    = model(sequences)
            probs     = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc   = accuracy_score(all_labels, all_preds)
    auc   = roc_auc_score(all_labels, all_probs)
    f1    = f1_score(all_labels, all_preds, zero_division=0)
    ece   = calculate_ece(all_probs, all_labels)
    cm    = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 45)
    print("          EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ECE       : {ece:.4f}  (lower = better calibrated)")
    print("-" * 45)
    print("  Confusion Matrix:")
    print(f"    TN={tn:>6}   FP={fp:>6}")
    print(f"    FN={fn:>6}   TP={tp:>6}")
    print("-" * 45)
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))
    print("=" * 45)


if __name__ == "__main__":
    main()