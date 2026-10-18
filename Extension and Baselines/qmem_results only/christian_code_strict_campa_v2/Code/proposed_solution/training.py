from utils import *
from proposed_solution.architecture.model import VQVAE2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as funct


# -------------------
#    Configuration
# -------------------
EPOCHS = 50
INPUT_DIR = real_path("/thesis/dataset/train")
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 8
IMG_SIZE = 512
CROP_SIZE = 256


# --------------
#    Training
# --------------
def main():
    # Ensure required directories exist
    ensure_dir([INPUT_DIR, CHECKPOINT_DIR])

    # Resize to 512x512 and convert to [0, 1] tensors
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    # Load the dataset
    dataset = datasets.ImageFolder(root = INPUT_DIR, transform = transform)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    # Load the model and optimizer
    model = VQVAE2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc = f"Epoch {epoch}/{EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(DEVICE)

            # Resize the batch to the model's expected crop input size
            full_input = funct.interpolate(imgs, size = (CROP_SIZE, CROP_SIZE), mode = 'bilinear')

            # Extract the 4 corner crops for each image in a single batch
            all_crops = torch.cat([get_corner_crops(img, CROP_SIZE) for img in imgs], dim = 0).to(DEVICE)

            # Forward pass
            full_recon, q_loss_full = model(full_input)
            crops_recon, q_loss_crops = model(all_crops)

            # Compare reconstructed outputs to original inputs
            loss_full = funct.mse_loss(full_recon, full_input)
            loss_crops = funct.mse_loss(crops_recon, all_crops)

            # Upsample the reconstruction
            full_recon_up = funct.interpolate(full_recon, size = (IMG_SIZE, IMG_SIZE), mode = 'bilinear')

            # Reassemble the high-res crop reconstructions
            crops_reassembled = reassemble_crops_to_full(crops_recon, IMG_SIZE)

            # Calculate MSE between the global structure and local details
            loss_consistency = funct.mse_loss(crops_reassembled, full_recon_up)

            # Total loss
            loss = loss_full + loss_crops + loss_consistency + q_loss_full + q_loss_crops

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track and display loss metrics
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Save model checkpoint
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
