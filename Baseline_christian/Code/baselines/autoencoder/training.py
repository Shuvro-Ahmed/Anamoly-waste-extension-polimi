from torch.utils.data import DataLoader
from tqdm import tqdm
from architecture.ae import UNetAE
from baselines.autoencoder.ae_utils.ae_utils import *
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path("/baselines/autoencoder/checkpoint")
EPOCHS = 50
BATCH_SIZE = 5
IMG_SIZE = 256
LR = 3e-4
FP_RATE = 0.01


# --------------
#    Training
# --------------
def main():
    # Ensure all required directories exist
    ensure_dir([OUTPUT_DIR])

    # Create the dataset and dataloader for training
    images = DatasetLoaderWithProcessing(train_path(), IMG_SIZE)
    training_loader = DataLoader(images, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, pin_memory = True)

    # Initialize model, optimizer, and mixed-precision scaler
    model = UNetAE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = 1e-4)
    scaler = torch.amp.GradScaler()

    # Define learning rate scheduler with warmup and cosine annealing
    warmup_epochs = max(1, int(0.1 * EPOCHS))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max(1, EPOCHS - warmup_epochs))

    # Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(training_loader, desc = f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            x = batch["image"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            mask = x[:, 3:4] if x.shape[1] > 3 else torch.ones_like(x[:, :1])

            # Forward pass (mixed precision)
            with torch.amp.autocast(device_type = DEVICE.type):
                pred = model(x)
                loss = masked_reconstruction_loss(pred, target, mask)

            # Backpropagation with gradient scaling and clipping
            optimizer.zero_grad(set_to_none = True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        # Scheduler step (after warmup)
        if epoch > warmup_epochs:
            scheduler.step()

        # Save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"u_net_epoch_{epoch}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)


if __name__ == "__main__":
    main()
