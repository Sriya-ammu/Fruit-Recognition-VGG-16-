import torch
from torch.optim.lr_scheduler import StepLR

# Dummy values for testing
best_val_acc = 0
best_val_loss = float('inf')

epochs_without_improvement = 0

# Example optimizer placeholder
optimizer = torch.optim.Adam(
    [torch.randn(1, requires_grad=True)],
    lr=0.001
)

# Learning rate scheduler
scheduler = StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

# Total epochs
N = 3

for epoch in range(N):

    print(f"\nEpoch {epoch+1}/{N}")

    # Dummy training results
    train_loss = 0.5 - (epoch * 0.05)

    # Dummy validation results
    val_loss = 0.4 + (epoch * 0.02)
    val_acc = 70 + (epoch * 2)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {"epoch": epoch},
            "best_model.pth"
        )

        print("Best model saved!")

    # Early stopping logic
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    # Stop if no improvement for 3 epochs
    if epochs_without_improvement >= 3:

        print("Early stopping triggered!")
        break

    # Update learning rate
    scheduler.step()

    print("Scheduler step completed")