from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Resize all images to same size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load training dataset
train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=transform
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Print detected classes
print("Classes:")
print(train_dataset.classes)

# Print total number of images
print("\nTotal Training Images:")
print(len(train_dataset))

# Load one batch
images, labels = next(iter(train_loader))

# Print shape
print("\nImage Batch Shape:")
print(images.shape)

print("\nLabels:")
print(labels)