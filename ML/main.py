import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Prepare the data
# Load the binary images and apply any preprocessing steps as needed
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((50, 50)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])


# 2. Define the CNN model
class NoiseFilterCNN(nn.Module):
    def __init__(self):
        super(NoiseFilterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 50 * 50)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 128 * 25 * 25)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# Define the loss function and optimizer
model = NoiseFilterCNN()
model = model.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters())

train_dataset = ImageFolder('./train_images/', transform=transform)
val_dataset = ImageFolder('./val_images/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the number of epochs and the batch size
num_epochs = 1000
batch_size = 32

# Loop over the number of epochs
for epoch in range(num_epochs):

    # Loop over the training data in batches
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= batch_size: break

        # Move the input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Round the inputs because they are inputted as 0.99 which makes me sad :C
        inputs = inputs.round()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        outputs = outputs.view(-1, 64)

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (i + 1) % 25 == 0:
            print("Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, batch_size,
                                                                    loss.item()))