import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import DeepfakeSequenceDataset
from dataloaders.transforms import get_train_transforms
from models.fusion import DeepfakeDetector
from utils.metrics import FocalLoss, calculate_metrics

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
        pin_memory
        =True
    )
    
    # Setup Model, Loss, Optimizer
    model = DeepfakeDetector(sequence_length=config['sequence_length']).to(device)
    criterion = FocalLoss(gamma=config['focal_gamma'])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

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
        
        # Epoch Metrics
        acc, auc = calculate_metrics(all_labels, all_probs)
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} - AUC: {auc:.4f}")

        # Save Checkpoint
        torch.save(model.state_dict(), f"weights/deepfake_model_ep{epoch+1}.pth")

if __name__ == "__main__":
    import os
    os.makedirs("weights", exist_ok=True)
    main()