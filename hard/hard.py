import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),                 
    transforms.Normalize((0.5,), (0.5,))   
])

trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


testset = datasets.MNIST('./data', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                
        self.fc1 = nn.Linear(16 * 14 * 14, 10)                            

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  
        x = x.view(-1, 16 * 14 * 14)             
        x = self.fc1(x)                            
        return x


model = SimpleCNN()
print(model)
print()


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
print(f"Training on: {device}")
print()


criterion = nn.CrossEntropyLoss()               
optimizer = optim.Adam(model.parameters(), lr=0.001) 


EPOCHS = 3

for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()         
        outputs = model(images)       
        loss = criterion(outputs, labels)  
        loss.backward()               
        optimizer.step()              

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")


model.eval() 
correct = 0
total = 0

with torch.no_grad(): 
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1) 

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")