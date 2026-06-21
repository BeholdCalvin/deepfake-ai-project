import os
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import DeepfakeSequenceDataset
from dataloaders.transforms import get_train_transforms, get_val_transforms
from models.fusion import DeepfakeDetector
from utils.metrics import FocalLoss, calculate_metrics


@torch.no_grad()
def _run_validation(model, val_loader, criterion, device):
    """Held-out evaluation pass. Returns (avg_loss, acc, auc)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        logits = model(sequences)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc, auc = calculate_metrics(all_labels, all_probs)
    return total_loss / max(len(val_loader), 1), acc, auc


def main():
    # Load Config
    with open("configs/train_ffpp.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Data
    train_transform = get_train_transforms(config['image_size'])
    train_dataset = DeepfakeSequenceDataset(
        data_dir=config['data_dir'],
        sequence_length=config['sequence_length'],
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True,
    )

    # Held-out validation set. This is what tells you whether the model is
    # actually learning to generalize, versus just memorizing the training
    # set — training-set accuracy alone (the old behavior) can look fine
    # while the model is useless on new data.
    val_loader = None
    val_data_dir = config.get('val_data_dir')
    if val_data_dir and os.path.isdir(val_data_dir):
        val_transform = get_val_transforms(config['image_size'])
        val_dataset = DeepfakeSequenceDataset(
            data_dir=val_data_dir,
            sequence_length=config['sequence_length'],
            transform=val_transform,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True,
            )
            print(f"[train] Validation set: {len(val_dataset)} sequences from {val_data_dir}")
        else:
            print(f"[train] WARNING: val_data_dir '{val_data_dir}' produced 0 sequences; "
                  f"training will proceed WITHOUT validation.")
    else:
        print(f"[train] WARNING: val_data_dir not found ('{val_data_dir}'); "
              f"training will proceed WITHOUT validation. You will not be able "
              f"to tell which checkpoint actually generalizes.")

    # Setup Model, Loss, Optimizer
    model = DeepfakeDetector(sequence_length=config['sequence_length']).to(device)
    criterion = FocalLoss(gamma=config['focal_gamma'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    os.makedirs("weights", exist_ok=True)
    best_val_auc = -1.0

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        all_labels = []
        all_probs = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for sequences, labels in loop:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients from LSTM/FFT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # Training-set metrics (optimistic; do not use these to judge the model)
        train_acc, train_auc = calculate_metrics(all_labels, all_probs)
        msg = (f"Epoch {epoch+1} - TrainLoss: {total_loss/len(train_loader):.4f} "
               f"- TrainAcc: {train_acc:.4f} - TrainAUC: {train_auc:.4f}")

        # Held-out validation metrics (use these to judge the model)
        if val_loader is not None:
            val_loss, val_acc, val_auc = _run_validation(model, val_loader, criterion, device)
            msg += f" | ValLoss: {val_loss:.4f} - ValAcc: {val_acc:.4f} - ValAUC: {val_auc:.4f}"

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), "weights/best_model.pth")
                msg += "  <- new best (saved to weights/best_model.pth)"

        print(msg)

        # Always keep the most recent checkpoint too, in case you want to
        # resume or inspect a specific epoch.
        torch.save(model.state_dict(), "weights/last_model.pth")

    if val_loader is None:
        print(
            "\n[train] No validation was run during training, so no "
            "'best_model.pth' was selected. Run evaluate.py against a "
            "held-out test set before trusting weights/last_model.pth."
        )


if __name__ == "__main__":
    main()