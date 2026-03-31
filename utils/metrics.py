import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Prevents nans
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def calculate_metrics(y_true, y_pred_probs):
    """Calculates accuracy and ROC-AUC"""
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    preds = (y_pred_probs >= 0.5).astype(int)
    accuracy = (preds == y_true).mean()
    
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5 # Happens if batch only has one class
        
    return accuracy, auc