import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
import pandas as pd
from scipy import interpolate
from torchsummary import summary

from Terrain import blobcheck

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}\n".format(device))


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


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


def main():
    loss_array = list()
    # print(summary(model, (1, 1, 256, 256)))

    if os.path.exists('./models/synthnav-model-0.pth'):
        print("Loaded Model")
        model.load_state_dict(torch.load(f='./models/synthnav-model-0.pth'))

    # Create an instance of the custom MSE loss function
    criterion = MAELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    def create_synced_dictionary(folder_name):
        sync_dir = {}
        for folder in os.walk(folder_name):
            for sub_folder in folder[1]:
                for file in os.listdir(os.path.join(folder_name, sub_folder)):
                    try:
                        name, image_type = file.split('_')
                    except:
                        return
                    if image_type == 'clean.jpeg':
                        # Add the clean image to the dictionary
                        sync_dir[name] = file
                    elif image_type == 'noisy.jpeg':
                            # Check if there is already a clean image for this pair in the dictionary
                        if name in sync_dir:
                            # Add the noisy image to the dictionary
                            sync_dir[name] = (sync_dir[name], file)
                        else:
                            # Add the noisy image to the dictionary with a placeholder for the clean image
                            sync_dir[name] = (None, file)

        return sync_dir

    dirname = os.path.dirname(__file__)

    high_quality = os.path.join(dirname, 'train_images')
    sync_dir = create_synced_dictionary('train_images')

    val_dir = os.path.join(dirname, 'val_images')
    val_sync = create_synced_dictionary(val_dir)

    # Training loop
    for epoch in range(num_epochs):
        i = 0
        for clean_image, noisy_image in sync_dir.values():
            if i >= batch_size:
                break
            # Load the clean and noisy images
            clean = plt.imread(os.path.join((high_quality + "/clean/"), clean_image))
            clean = np.where(clean >= 128, 1.0, 0.0)
            noisy = plt.imread(os.path.join((high_quality + "/noisy/"), noisy_image))
            noisy = np.where(noisy >= 128, 1.0, 0.0)

            noisy = torch.tensor(noisy).view(1, 1, 256, 256)
            clean = torch.tensor(clean).view(1, 1, 256, 256).to(device)

            noisy = noisy.type(torch.cuda.FloatTensor)
            clean = clean.type(torch.cuda.FloatTensor)

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

            if (i + 1) % batch_size == 0:
                print("Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, batch_size,
                                                                        loss.item()))
            i += 1

        if (epoch + 1) % 20 == 0:
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
    print("----- Training Done -----")
    print("----- Visual Starting -----")
    visualize_predictions(model, val_sync, loss_array, val_dir)
    loss_graph(loss_array)


def visualize_predictions(model, val_sync, loss_array, val_dir):
    # Iterate over the clean-noisy image pairs in the dictionary
    vis_iter = 0
    for clean_image, noisy_image in val_sync.values():
        if vis_iter == 1:
            break
        # Load the clean and noisy images
        clean = plt.imread(os.path.join((val_dir + "/clean/"), clean_image))
        clean = np.where(clean >= 128, 1.0, 0.0)
        noisy = plt.imread(os.path.join((val_dir + "/noisy/"), noisy_image))
        noisy = np.where(noisy >= 128, 1.0, 0.0)

        noisy = torch.tensor(noisy).view(1, 1, 256, 256)
        clean = torch.tensor(clean).view(1, 1, 256, 256).to(device)

        noisy_image = noisy.type(torch.cuda.FloatTensor)
        clean_image = clean.type(torch.cuda.FloatTensor)

        # Use the model to predict the de-noised image
        de_noised_image = model(noisy_image)
        de_noised_image = de_noised_image.view(256, 256)
        de_noised_image = de_noised_image.cpu()
        de_noised_image = de_noised_image.detach().numpy()

        # Use the model on the clean image
        clean_deionised = model(clean_image)
        clean_deionised = clean_deionised.view(256, 256)
        clean_deionised = clean_deionised.cpu()
        clean_deionised = clean_deionised.detach().numpy()

        clean_image = clean_image.view(256, 256)
        clean_image = clean_image.cpu()
        clean_image = clean_image.detach().numpy()

        noisy_image = noisy_image.view(256, 256)
        noisy_image = noisy_image.cpu()
        noisy_image = noisy_image.detach().numpy()

        cmap = 'plasma_r'

        # Visualize the clean, noisy, and de-noised images
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].imshow(clean_image, cmap=cmap)
        axs[0, 0].set_title('Input (Clean)')
        axs[1, 0].imshow(clean_deionised, cmap=cmap)
        axs[1, 0].set_title('Output (Clean)')
        axs[0, 1].imshow(noisy_image, cmap=cmap)
        axs[0, 1].set_title('Input (Noisy)')
        axs[1, 1].imshow(de_noised_image, cmap=cmap)
        axs[1, 1].set_title('Output (Noisy)')
        axs[0, 2].imshow(clean_image - noisy_image, cmap=cmap)
        axs[0, 2].set_title('Input Difference')
        axs[1, 2].imshow(clean_deionised - de_noised_image, cmap=cmap)
        axs[1, 2].set_title('Output Difference')
        plt.show()
        vis_iter = 1


def loss_graph(loss_array):
    x = np.arange(0, batch_size * num_epochs)
    y = loss_array

    plt.title("Loss graph")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.plot(x, y, color="green")


if __name__ == "__main__":
    # Define the number of epochs and the batch size
    is_int = False
    while not is_int:
        try:
            num_epochs = int(input("How many epochs: "))
            is_int = True
        except:
            is_int = False
            print("Please provide a integer value")

    count = 0

    # Iterate directory
    dirname = os.path.dirname(__file__)

    high_quality = os.path.join(dirname, 'train_images/noisy')

    for path in os.listdir(high_quality):
        # check if current path is a file
        if os.path.isfile(os.path.join('train_images/noisy', path)):
            count += 1

    batch_size = 2
    model = EncoderDecoder().to(device)

    main()
