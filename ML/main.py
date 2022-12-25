import os

import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


# Define the CNN model
class NoiseFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 7 * 7, 1)

    def forward(self, for_x):
        for_x = self.pool(torch.relu(self.conv1(for_x)))
        for_x = self.pool(torch.relu(self.conv2(for_x)))
        for_x = for_x.view(-1, 64 * 7 * 7)
        for_x = torch.sigmoid(self.fc(for_x))
        return for_x


class ImageDataset(Dataset):
    def __init__(self, root_dir, init_transform=None):
        self.root_dir = root_dir
        self.transform = init_transform
        self.images = []
        self.labels = []

        # Load the image and label data
        for filename in os.listdir(root_dir):
            # Extract the label from the filename (e.g., "1_image.jpg" -> 1)
            label = int(filename.split("_")[0])
            self.labels.append(label)
            self.images.append(os.path.join(root_dir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        get_image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            get_image = self.transform(get_image)
        return get_image, label


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageDataset("clean/", init_transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model and the optimizer
model = NoiseFilter()
optimizer = optim.Adam(model.parameters())

num_epochs = 1000

# Train the model
for epoch in range(num_epochs):
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = nn.BCEWithLogitsLoss()(y_pred, y)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    for x, y in dataloader:
        y_pred = model(x)
        acc = (y_pred > 0.5).eq(y).float().mean()

# Use the model to filter noise from a new image
image = ...  # your binary image with noise
image = image.unsqueeze(0).unsqueeze(0)  # add dummy batch and channel dimensions
image = image.to(device)  # move image to the device (e.g., GPU) if available
model.eval()
with torch.no_grad():
    y_pred = model(image).squeeze().cpu().numpy()
image_clean = (y_pred > 0.5).astype(int)  # filter out pixels predicted to be noise

print(image_clean)
