import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}\n".format(device))


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.clamp(x, min=0, max=1)
        return x


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


def saveModel(model, model_name):
    torch.save(model.state_dict(), "./models/{}".format(model_name))
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save("./models/synthnav-model-script.pt")


def main():
    loss_array = list()

    # Only load the model if it already exists
    if os.path.exists("./models/{}".format(model_name)):
        print("Loaded Model")
        model.load_state_dict(torch.load(f="./models/{}".format(model_name)))

    # Create an instance of the custom MSE loss function
    criterion = MAELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def create_synced_dictionary(folder_name):
        sync_dir_inner = {}
        for folder in os.walk(folder_name):
            for sub_folder in folder[1]:
                for file in os.listdir(os.path.join(folder_name, sub_folder)):
                    try:
                        name, image_type = file.split("_")
                    except ValueError:
                        continue

                    if image_type == "clean.jpeg":
                        # Add the clean image to the dictionary
                        sync_dir_inner[name] = file
                    elif image_type == "noisy.jpeg":
                        # Check if there is already a clean image for this pair in the dictionary
                        if name in sync_dir_inner:
                            # Add the noisy image to the dictionary
                            sync_dir_inner[name] = (sync_dir_inner[name], file)
                        else:
                            # Add the noisy image to the dictionary with a placeholder for the clean image
                            sync_dir_inner[name] = (None, file)

        return sync_dir_inner

    dirname = os.path.dirname(__file__)

    high_quality = os.path.join(dirname, "train_images")
    sync_dir = create_synced_dictionary("train_images")

    val_dir = os.path.join(dirname, "val_images")
    val_sync = create_synced_dictionary(val_dir)

    # Training loop
    for epoch in range(num_epochs):
        i = 0
        model.train()
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
                print(
                    "Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}%".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        batch_size,
                        (1 - loss.item()) * 100,
                    )
                )
            i += 1

        if (epoch + 1) % 20 == 0:
            saveModel(model, model_name)

    loss_array = np.array(loss_array)

    saveModel(model, model_name)
    print("----- Training Done -----")
    print("----- Visual Starting -----")
    visualize_predictions(model, val_sync, val_dir)


def visualize_predictions(model, val_sync, val_dir):
    model.eval()
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

        cmap = "plasma_r"

        # Visualize the clean, noisy, and de-noised images
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].imshow(clean_image, cmap=cmap)
        axs[0, 0].set_title("Input (Clean)")
        axs[1, 0].imshow(clean_deionised, cmap=cmap)
        axs[1, 0].set_title("Output (Clean)")
        axs[0, 1].imshow(noisy_image, cmap=cmap)
        axs[0, 1].set_title("Input (Noisy)")
        axs[1, 1].imshow(de_noised_image, cmap=cmap)
        axs[1, 1].set_title("Output (Noisy)")
        axs[0, 2].imshow(clean_image - noisy_image, cmap=cmap)
        axs[0, 2].set_title("Input Difference")
        axs[1, 2].imshow(clean_deionised - de_noised_image, cmap=cmap)
        axs[1, 2].set_title("Output Difference")
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

    model = EncoderDecoder().to(device)
    print(torchsummary.summary(model, (1, 256, 256)))

    is_int = False
    while not is_int:
        try:
            num_epochs = int(input("How many epochs: "))
            is_int = True
        except ValueError:
            is_int = False
            print("Please provide a integer value")

    count = 0

    # Iterate directory
    dirname = os.path.dirname(__file__)

    high_quality = os.path.join(dirname, "train_images/noisy")
    model_name = "synthnav-model-0.pth"
    for path in os.listdir(high_quality):
        # check if current path is a file
        if os.path.isfile(os.path.join("train_images/noisy", path)):
            count += 1

    batch_size = 128
    main()
