import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data
from torch.utils.data import Subset, TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Prepare the data
# Load the binary images and apply any preprocessing steps as needed
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((50, 50)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale()
])


# 2. Define the CNN model
class NoiseFilterCNN(nn.Module):
    def __init__(self):
        super(NoiseFilterCNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Define the max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 48 * 48, 128)
        print(self.fc1)
        self.fc2 = nn.Linear(128, 1)

        # Define the activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the convolutional layers and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        print(x.shape)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 64 * 48 * 48)

        print(x.shape, x)
        # Apply the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = NoiseFilterCNN()
model = model.to(device)

train_dataset = ImageFolder('./train_images/', transform=transform)
val_dataset = ImageFolder('./val_images/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

num_epochs = 100

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(num_epochs):
    # Iterate over the training data
    for inputs, labels in train_loader:
        # Move the data to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation data
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = correct / total
        print(f'Epoch {epoch + 1}: Validation accuracy = {val_acc:.2f}')
