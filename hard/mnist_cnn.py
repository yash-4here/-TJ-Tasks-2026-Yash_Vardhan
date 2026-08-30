"""
Level 3 Task: Simple CNN for MNIST digit classification (PyTorch)
Run this top to bottom in Google Colab (Runtime > Change runtime type > GPU is optional, CPU works fine for this small model).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. LOAD AND NORMALIZE DATA
# ---------------------------------------------------------
# ToTensor() converts images to PyTorch tensors AND scales pixel
# values from [0, 255] down to [0, 1] automatically.
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
testset = datasets.MNIST('./data', download=True, train=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 2. DEFINE THE CNN ARCHITECTURE
# ---------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv2D: 1 input channel (grayscale), 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves image size: 28x28 -> 14x14
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, 10)  # 10 output classes: digits 0-9

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x  # raw scores ("logits") for each of the 10 digit classes

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
print(model)

# ---------------------------------------------------------
# 3. TRAIN THE MODEL
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()          # good default loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 3  # one epoch = one full pass through the training data

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # reset gradients from last step
        outputs = model(images)        # forward pass: get predictions
        loss = criterion(outputs, labels)  # compare predictions to true labels
        loss.backward()                # backward pass: compute gradients
        optimizer.step()               # update model weights

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training loss: {avg_loss:.4f}")

# ---------------------------------------------------------
# 4. EVALUATE ON TEST DATA
# ---------------------------------------------------------
model.eval()  # switch to evaluation mode (disables things like dropout, if used)
correct = 0
total = 0

with torch.no_grad():  # no need to track gradients during evaluation
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # pick the class with the highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
