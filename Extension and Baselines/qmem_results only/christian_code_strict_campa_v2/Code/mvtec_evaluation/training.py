from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as funct


# -------------------
#    Configuration
# -------------------
EPOCHS = 50
INPUT_DIR = real_path("/dataset_mvtec")
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 8
IMG_SIZE = 512


# --------------
#    Training
# --------------
def main(category_name):
    # Set up directories and paths
    category_path = os.path.join(INPUT_DIR, category_name, "train")
    cat_checkpoint_dir = os.path.join(CHECKPOINT_DIR, category_name)
    ensure_dir([cat_checkpoint_dir])

    # Standardize size and normalize pixels
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    # Load only the normal training images for this category
    dataset = datasets.ImageFolder(root = category_path, transform = transform)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    # Initialize model and optimizer
    model = VQVAE2(num_embeddings = 256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc = f"{category_name.capitalize()} - Epoch {epoch}/{EPOCHS}")
        for imgs, _ in pbar:
            imgs = imgs.to(DEVICE)

            # Resize full images
            full_input = funct.interpolate(imgs, size = (IMG_SIZE, IMG_SIZE), mode = 'bilinear')

            # Extract corner crop
            all_crops = torch.cat([get_corner_crops(img, IMG_SIZE) for img in imgs], dim = 0).to(DEVICE)

            # Forward pass
            full_recon, q_loss_full = model(full_input)
            crops_recon, q_loss_crops = model(all_crops)

            # Compute reconstruction error
            loss_full = funct.mse_loss(full_recon, full_input)
            loss_crops = funct.mse_loss(crops_recon, all_crops)

            # Multi-scale consistency loss
            full_recon_up = funct.interpolate(full_recon, size = (IMG_SIZE, IMG_SIZE), mode = 'bilinear')
            crops_reassembled = reassemble_crops_to_full(crops_recon, IMG_SIZE)
            loss_consistency = funct.mse_loss(crops_reassembled, full_recon_up)

            # Total loss
            loss = loss_full + loss_crops + loss_consistency + q_loss_full + q_loss_crops

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Save intermediate checkpoint
        torch.save(model.state_dict(), f"{cat_checkpoint_dir}/epoch_{epoch}.pth")

    # Save the final model for inference
    ensure_dir([f"checkpoint/{category_name}"])
    torch.save(model.state_dict(), f"checkpoint/{category_name}/model.pth")


if __name__ == "__main__":
    # Detect all category folders
    categories = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

    for category in categories:
        main(category)
