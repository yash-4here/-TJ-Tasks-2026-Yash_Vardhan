"""
Level 3 Task: Building a Basic Convolutional Neural Network (CNN)
------------------------------------------------------------------
Trains a small CNN on the MNIST handwritten digit dataset using PyTorch.
Run this in Google Colab (with GPU runtime) for fastest training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# STEP 1: Load & Normalize Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                  # scales pixels from [0,255] -> [0,1]
    transforms.Normalize((0.5,), (0.5,))    # rescales [0,1] -> [-1,1] for stable training
])

# Training set - the model learns from these 60,000 images
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Test set - 10,000 images the model NEVER trains on, used only to check accuracy
testset = datasets.MNIST('./data', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# -----------------------------
# STEP 2: Define the CNN Architecture
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 1 input channel -> 16 feature maps
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # halves width & height
        self.fc1 = nn.Linear(16 * 14 * 14, 10)                             # final decision layer -> 10 digit classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # conv -> ReLU -> pool
        x = x.view(-1, 16 * 14 * 14)               # flatten into a 1D list per image
        x = self.fc1(x)                            # dense layer -> 10 raw scores
        return x


model = SimpleCNN()
print(model)
print()

# Use the best available device:
# - "cuda"  -> NVIDIA GPU (Colab, most Windows/Linux machines with an NVIDIA card)
# - "mps"   -> Apple Silicon GPU (M1/M2/M3/M4 Macs, via Metal Performance Shaders)
# - "cpu"   -> fallback if neither GPU backend is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
print(f"Training on: {device}")
print()


# -----------------------------
# STEP 3: Loss Function & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()               # measures how wrong the predictions are
optimizer = optim.Adam(model.parameters(), lr=0.001)  # updates weights to reduce that error


# -----------------------------
# STEP 4: Training Loop
# -----------------------------
EPOCHS = 3

for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # forward pass -> 10 scores per image
        loss = criterion(outputs, labels)  # compare predictions to true labels
        loss.backward()                # backward pass -> compute gradients
        optimizer.step()               # update weights using those gradients

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")


# -----------------------------
# STEP 5: Evaluate on the Test Set
# -----------------------------
model.eval()  # puts the model in "evaluation mode" (affects certain layers, good practice)
correct = 0
total = 0

with torch.no_grad():  # no need to track gradients when we're just checking accuracy
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)  # pick the digit with the highest score

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")