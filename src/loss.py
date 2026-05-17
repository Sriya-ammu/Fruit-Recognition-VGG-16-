import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Formula: FL = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        alpha (float): Balancing factor. Scales the overall loss magnitude.
                       Typical value: 0.25 or 1.0
        gamma (float): Focusing parameter. Higher gamma reduces loss for
                       easy (high-confidence) examples, forcing the model
                       to focus on hard ones.
                       gamma=0 → identical to standard cross-entropy
                       Typical value: 2.0
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()          # Required: initializes nn.Module internals
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs : raw logits,  shape (N, C)  — NOT softmaxed yet
            labels  : integer class indices, shape (N,)

        Returns:
            Scalar loss tensor
        """
        # Step 1 — Convert logits → log-probabilities (numerically stable)
        # F.log_softmax does log(softmax(x)) in one fused, stable operation
        log_probs = F.log_softmax(outputs, dim=1)           # shape: (N, C)

        # Step 2 — Pick log(pt): the log-prob of the TRUE class for each sample
        # gather() selects one value per row using the label index
        log_pt = log_probs.gather(dim=1, index=labels.unsqueeze(1))  # (N, 1)
        log_pt = log_pt.squeeze(1)                                    # (N,)

        # Step 3 — Recover pt = exp(log(pt))  [kept in log-space until needed]
        pt = log_pt.exp()                                             # (N,)

        # Step 4 — Apply the focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma                      # (N,)

        # Step 5 — Full formula: FL = -alpha * (1-pt)^gamma * log(pt)
        focal_loss = -self.alpha * focal_weight * log_pt              # (N,)

        # Return mean loss over the batch (same convention as nn.CrossEntropyLoss)
        return focal_loss.mean()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # Dummy data: batch of 8, 15 classes
    outputs = torch.randn(8, 15)
    labels  = torch.randint(0, 15, (8,))

    # Focal Loss
    focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss_val  = focal_criterion(outputs, labels)
    print(f"Focal Loss       : {focal_loss_val.item():.4f}")

    # Standard Cross-Entropy (for comparison)
    standard_criterion = nn.CrossEntropyLoss()
    standard_loss_val  = standard_criterion(outputs, labels)
    print(f"Cross-Entropy    : {standard_loss_val.item():.4f}")

    # Sanity check: with gamma=0, Focal Loss ≈ alpha * CrossEntropy
    focal_gamma0 = FocalLoss(alpha=1.0, gamma=0.0)(outputs, labels)
    print(f"Focal(α=1,γ=0)   : {focal_gamma0.item():.4f}  ← should match CE above")
