import torch
import torch.nn as nn

from src.model import build_model
from src.dataset import train_loader

def train_one_epoch(model,loader,loss_fn,optimizer,device):
    """
    Runs one full pass over the training set
    
    Steps per batch:
    1. zero_grad - clear accumulated gradients from previous batch
    2. forward - get model predictions
    3. loss - compute loss between predictions and true labels
    4. backward - compute gradients via backprop
    5. step - update weights using those grads
    
    return - avg_loss(float) - mean loss across all batches in this epoch
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item() # .item pulls plain float from tensor

        avg_loss = running_loss/len(loader)
        return avg_loss
    
# exp 1 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    EPOCHS = 3
    LR = 1e-4

    model = build_model(num_classes=15).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    exp1_train_losses = []

    for epoch in range(1,EPOCHS+1):
        avg_loss = train_one_epoch(model,train_loader,loss_fn,optimizer,device)
        exp1_train_losses.append(round(avg_loss, 4))
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

    print(f"\nexp1_train_losses = {exp1_train_losses}")