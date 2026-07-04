from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from architecture.vq_ae import VQVAE
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path("/baselines/vector_quantized_autoencoder/checkpoint")
EPOCHS = 50
IMG_SIZE = 256


# --------------
#    Training
# --------------
def main():
    # Ensure the necessary directories exist
    ensure_dir([OUTPUT_DIR])

    # Preprocessing
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    # Load dataset
    dataset = datasets.ImageFolder(root = train_path(), transform = transform)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, num_workers = 2)

    # Initialize the model, move it to the target compute device, and initialize the optimizer
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for imgs, _ in tqdm(dataloader, desc = f"Epoch {epoch}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)

            # Forward pass
            x_recon, q_loss = model(imgs)

            # Combined loss computation
            loss = functional.mse_loss(x_recon, imgs) + q_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save the model checkpoint
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/epoch_{epoch}.pth")


if __name__ == '__main__':
    main()
