import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image
from torchsummary import summary

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
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = x.to(device)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv10 = nn.Conv2d(16, 1, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.upsample(x)
        x = self.conv7(x)
        x = self.upsample(x)
        x = self.conv8(x)
        x = self.upsample(x)
        x = self.conv9(x)
        x = self.upsample(x)
        x = self.conv10(x)
        return x


# Define the encoder-decoder model
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomMSE(nn.Module):
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, de_noised_image, clean_image):
        # Calculate the element-wise difference between the clean image and the de-noised image
        diff = clean_image - de_noised_image

        # Calculate the mean squared error by taking the mean of the squared difference
        mse = torch.mean(diff ** 2)

        return mse


# Define the loss function and optimizer
model = EncoderDecoder().to(device)

print(summary(model, (1, 256, 256)))

if os.path.exists('./models/synthnav-model-0.pth'):
    print("Loaded Model")
    model.load_state_dict(torch.load(f='./models/synthnav-model-0.pth'))

# Create an instance of the custom MSE loss function
criterion = CustomMSE()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the number of epochs and the batch size
num_epochs = 100
batch_size = 64

loss_array = list()


def create_synced_dictionary(root_dir):
    # Create a dictionary to store the clean-noisy image pairs
    synced_dict = {}

    # Iterate over the files in the root directory
    for file in os.listdir(root_dir):
        # Split the file name into the image name and the image type (clean or noisy)
        name, image_type = file.split('_')
        # Check if the image type is clean or noisy
        if image_type == 'clean.png':
            # Add the clean image to the dictionary
            synced_dict[name] = file
        elif image_type == 'noisy.png':
            # Check if there is already a clean image for this pair in the dictionary
            if name in synced_dict:
                # Add the noisy image to the dictionary
                synced_dict[name] = (synced_dict[name], file)
            else:
                # Add the noisy image to the dictionary with a placeholder for the clean image
                synced_dict[name] = (None, file)

    return synced_dict


root_dir = 'train_images'
sync_dir = create_synced_dictionary(root_dir)
val_dir = 'val_images'
val_sync = create_synced_dictionary(val_dir)

# Training loop
for epoch in range(num_epochs):
    i = 0
    for clean_image, noisy_image in sync_dir.values():
        if i >= batch_size:
            break
        # Load the clean and noisy images
        clean = plt.imread(os.path.join(root_dir, clean_image))
        noisy = plt.imread(os.path.join(root_dir, noisy_image))

        noisy = torch.tensor(noisy).view(1, 1, 256, 256)
        clean = torch.tensor(clean).view(1, 1, 256, 256).to(device)

        # Forward pass
        output = model(noisy)

        # Calculate loss
        loss = criterion(output, clean)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad()

        loss_array.append(loss.item())

        if (i + 1) % 32 == 0:
            print("Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, batch_size,
                                                                    loss.item()))
        i += 1

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

# model = model.to("cpu")

loss_array = np.array(loss_array)

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


def visualize_predictions():
    # Iterate over the clean-noisy image pairs in the dictionary
    iter = 0
    for clean_image, noisy_image in val_sync.values():
        if iter == 1:
            break
        # Load the clean and noisy images
        clean_image = plt.imread(os.path.join(val_dir, clean_image))
        noisy_image = plt.imread(os.path.join(val_dir, noisy_image))

        # Use the model to predict the de-noised image
        de_noised_image = model(torch.tensor(noisy_image).view(1, 1, 256, 256))
        de_noised_image = de_noised_image.view(256, 256)
        de_noised_image = de_noised_image.cpu()
        de_noised_image = de_noised_image.detach().numpy()

        # Visualize the clean, noisy, and de-noised images
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(clean_image)
        ax[0].set_title('Clean image')
        ax[1].imshow(noisy_image)
        ax[1].set_title('Noisy image')
        ax[2].imshow(de_noised_image)
        ax[2].set_title('De-noised image')
        plt.show()
        iter = 1


def loss_graph():
    x = np.arange(0, batch_size * num_epochs)
    y = loss_array

    plt.title("Loss graph")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.plot(x, y, color="green")
    plt.show()


# Test the model visualizer
# visualize_model(model, sync_dir[1][0], sync_dir[1][1])
visualize_predictions()
loss_graph()
