import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}\n".format(device))
# 1. Prepare the data
# Load the binary images and apply any preprocessing steps as needed
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((50, 50)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])


# 2. Define the CNN model
# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, 3, padding=2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.upsample(x)
        x = self.conv6(x)
        return x


# Define the encoder-decoder model
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.to(device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomMSE(torch.nn.Module):
    def __init__(self, clean_images):
        super(CustomMSE, self).__init__()
        self.clean_images = clean_images.to(device)

    def forward(self, output):
        # Calculate MSE loss between output and clean images
        loss = torch.nn.functional.mse_loss(output, self.clean_images)
        return loss


# Define the loss function and optimizer
model = EncoderDecoder().to(device)

print(summary(model, (1, 50, 50)))

if os.path.exists('./models/synthnav-model-0.pth'):
    print("Loaded Model")
    model.load_state_dict(torch.load(f='./models/synthnav-model-0.pth'))

train_dataset = ImageFolder('./train_images/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

filenames = os.listdir('./val_images/clean/')

clean_image_arrays = []
noise_image_arrays = []

for filename in filenames:
    # Open image file
    cleanImage = Image.open('./val_images/clean/' + filename)
    noiseImage = Image.open('./val_images/noise/' + filename)

    # Convert image to numpy array
    clean_image_array = np.array(cleanImage)
    noise_image_array = np.array(noiseImage)
    # Append image array to list
    clean_image_arrays.append(clean_image_array)
    noise_image_arrays.append(noise_image_array)

# Convert list of image arrays to a single numpy array
clean_images_numpy = np.stack(clean_image_arrays, axis=0)
noise_images_numpy = np.stack(noise_image_arrays, axis=0)

# Load clean images and convert them to a PyTorch tensor
clean_images = torch.from_numpy(clean_images_numpy).float()
noise_images = torch.from_numpy(noise_images_numpy).float()

clean_images = clean_images[:64].view(64, 1, 50, 50)
noise_images = noise_images[:64].view(64, 1, 50, 50)

# Create an instance of the custom MSE loss function
loss_fn = CustomMSE(clean_images)

criterion = CustomMSE(clean_images)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the number of epochs and the batch size
num_epochs = 2
batch_size = 75

loss_array = list()

# Training loop
for epoch in range(num_epochs):
    for i, (inner_input, target) in enumerate(train_loader):
        if i >= batch_size:
            break
        # Forward pass
        output = model(inner_input)

        # Calculate loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad()

        loss_array.append(loss.item())

        if (i + 1) % 25 == 0:
            print("Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, batch_size,
                                                                    loss.item()))

    if (epoch + 1) % 10 == 0:
        # 1. Create models directory
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        # 2. Create model save path
        MODEL_NAME = "synthnav-model-0.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        # 3. Save the model state dict
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model.state_dict(),  # only saving the state_dict() only saves the models learned parameters
                   f=MODEL_SAVE_PATH)

model = model.to("cpu")

loss_array = np.array(loss_array)

for i, (inputs, labels) in enumerate(train_loader):
    inputs = inputs.to("cpu")
    labels = labels.to("cpu")

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "synthnav-model-0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),  # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)


def visualize_model(model, clean_images, noisy_images):
    # Predict output for clean and noisy images
    model = model.to(device)

    clean_output = model(clean_images)
    noisy_output = model(noisy_images)

    clean_output, noisy_output = clean_output.detach(), noisy_output.detach(),

    clean_output, noisy_output, model = clean_output.to("cpu"), noisy_output.to("cpu"), model.to("cpu")

    clean_images = clean_images.squeeze(1)
    clean_output = clean_output.squeeze(1)
    noisy_images = noisy_images.squeeze(1)
    noisy_output = noisy_output.squeeze(1)

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].imshow(clean_images[0], cmap='gray')
    axs[0, 0].set_title('Input (Clean)')
    axs[0, 1].imshow(clean_output[0], cmap='gray')
    axs[0, 1].set_title('Output (Clean)')
    axs[1, 0].imshow(noisy_images[0], cmap='gray')
    axs[1, 0].set_title('Input (Noisy)')
    axs[1, 1].imshow(noisy_output[0], cmap='gray')
    axs[1, 1].set_title('Output (Noisy)')
    axs[2, 0].imshow(clean_images[0] - noisy_images[0], cmap='gray')
    axs[2, 0].set_title('Input Difference')
    axs[2, 1].imshow(clean_output[0] - noisy_output[0], cmap='gray')
    axs[2, 1].set_title('Output Difference')
    plt.show()


def loss_graph():
    x = np.arange(0, batch_size * num_epochs)
    y = loss_array

    plt.title("Loss graph")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.plot(x, y, color="green")
    plt.show()


# Test the model visualizer
visualize_model(model, clean_images, noise_images)
loss_graph()
