from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from architecture.vq_ae_2 import VQVAE2
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path("/baselines/vector_quantized_autoencoder_2/checkpoint")
EPOCHS = 50
IMG_SIZE = 256


# --------------
#    Training
# --------------
def main():
    # Ensure the necessary directories exist
    ensure_dir([OUTPUT_DIR])

    # Random crop to 256x256 and convert to tensor
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Load dataset from INPUT_DIR, applying the defined transformations and wrap dataset in a DataLoader
    dataset = datasets.ImageFolder(root = train_path(), transform = transform)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, num_workers = 2)

    # Initialize the model and move it to the selected device and initialize the optimizer
    model = VQVAE2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    # Main training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        # Iterate over batches
        for imgs, _ in tqdm(dataloader, desc = f"Epoch {epoch}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            x_recon, q_loss = model(imgs)
            loss = functional.mse_loss(x_recon, imgs) + q_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/epoch_{epoch}.pth")


if __name__ == '__main__':
    main()
