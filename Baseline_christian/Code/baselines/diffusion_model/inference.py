import time
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import roc_auc_score
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path('/baselines/diffusion_model/output')
CHECKPOINT_DIR = real_path('/baselines/diffusion_model/checkpoint/unet')
IMG_SIZE = 256


# ---------------
#    Inference
# ---------------
def main():
    # Ensure all required directories exist
    ensure_dir([OUTPUT_DIR, CHECKPOINT_DIR])

    # Image preprocessing: resize and convert to tensor
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    # Load the pretrained UNet diffusion model
    unet = UNet2DModel.from_pretrained(CHECKPOINT_DIR).to(DEVICE)

    # Define the DDPM scheduler used during reverse diffusion
    sched = DDPMScheduler(num_train_timesteps = 1000, beta_start = 0.0001, beta_end = 0.02, beta_schedule = "linear")

    # Set the model to evaluation mode
    unet.eval()

    # Containers for global evaluation metrics
    pixel_scores_all, pixel_labels_all, iou_list, inference_times = [], [], [], []

    # Load all test images and sort them naturally
    image_paths = load_test_images()

    # Process each image
    for img_path in tqdm(image_paths, desc = "Processing images"):
        # Load image and corresponding mask
        mask_path, name = load_test_masks(img_path)

        start_time = time.time()

        # Preprocess image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np = (transform(mask).squeeze().numpy() > 0.5).astype(np.uint8)

        # Resize the image and convert to numpy
        x_vis = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
        x_np = x_vis.astype(np.float32)

        # Normalize the image to [-1, 1] for the diffusion model
        img_tensor = torch.from_numpy(x_np.transpose(2, 0, 1) / 255.0 * 2 - 1).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            # Start from pure Gaussian noise
            noise = torch.randn_like(img_tensor)

            # Set fewer steps for faster inference
            sched.set_timesteps(50)
            sample = noise

            # Reverse diffusion process
            for t in sched.timesteps:
                noise_pred = unet(sample, t).sample
                sample = sched.step(noise_pred, t, sample).prev_sample

        # Convert the reconstructed image back to [0, 255]
        recon_np = ((sample[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.float32)

        # Pixel-wise reconstruction error with smoothing
        err_map = np.mean(np.abs(x_np - recon_np), axis = 2)
        err_map = gaussian_filter(err_map, sigma = 1)

        # Threshold error map using the 90th percentile
        thr = np.percentile(err_map, 90)
        predicted_mask = (err_map >= thr).astype(np.uint8)

        # Remove small connected components
        labeled_mask, num_features = label(predicted_mask)
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < 100:
                predicted_mask[labeled_mask == i] = 0

        # Record inference time
        inference_times.append(time.time() - start_time)

        # Accumulate pixel-wise scores and labels
        pixel_scores_all.append(err_map.flatten())
        pixel_labels_all.append(mask_np.flatten())

        # Compute image-level IoU
        iou = compute_iou(predicted_mask, mask_np)
        iou_list.append(iou)

        # Save visualization results
        visualize(x_vis, recon_np, predicted_mask, mask_np, iou, OUTPUT_DIR, name)

    # Final evaluation metrics computation
    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)
    auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
    mean_iou = np.mean(iou_list).astype(float)
    avg_time = np.mean(inference_times).astype(float)

    # Print final results
    print_results(auroc, mean_iou, avg_time)


if __name__ == "__main__":
    main()
