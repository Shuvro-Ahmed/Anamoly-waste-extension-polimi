import time
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from architecture.vq_ae_2 import VQVAE2
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path('/baselines/vector_quantized_autoencoder_2/output')
CHECKPOINT_DIR = real_path('/baselines/vector_quantized_autoencoder_2/checkpoint')
IMG_SIZE = 256


# ---------------
#    Inference
# ---------------
def main():
    # Ensure all required directories exist
    ensure_dir([OUTPUT_DIR, CHECKPOINT_DIR])

    # Standardized preprocessing: resize and convert to tensor [0, 1]
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    # Load the VQ-VAE-2 model
    model = VQVAE2().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pth")

    # Load state dict with weights_only=True for security/future-proofing
    model.load_state_dict(torch.load(checkpoint_path, map_location = DEVICE, weights_only = True))
    model.eval()

    # Containers for global evaluation metrics
    pixel_scores_all, pixel_labels_all, iou_list, inference_times = [], [], [], []

    # Load test images and sort them naturally
    image_paths = load_test_images()

    # Process each image
    for img_path in tqdm(image_paths, desc = "Processing images"):
        mask_path, name = load_test_masks(img_path)

        start_time = time.time()

        # Load image and ground truth mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np = (np.array(mask.resize((IMG_SIZE, IMG_SIZE))) > 127).astype(np.uint8)

        # Prepare input tensor and visualization reference
        x_vis = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            recon, _ = model(img_tensor)

        # Convert reconstruction to [0, 255] numpy for error calculation
        recon_np = (recon.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)
        x_np = x_vis.astype(np.float32)

        # Pixel-wise reconstruction error with Gaussian smoothing
        err_map = np.mean(np.abs(x_np - recon_np), axis = 2)
        err_map = gaussian_filter(err_map, sigma = 1)

        # Threshold error map using the 90th percentile (Matching Diffusion Baseline)
        thr = np.percentile(err_map, 90)
        predicted_mask = (err_map >= thr).astype(np.uint8)

        # Morphological cleaning: Remove small noisy components (< 100 pixels)
        labeled_mask, num_features = label(predicted_mask)
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < 100:
                predicted_mask[labeled_mask == i] = 0

        # Record inference time
        inference_times.append(time.time() - start_time)

        # Accumulate metrics data
        pixel_scores_all.append(err_map.flatten())
        pixel_labels_all.append(mask_np.flatten())

        # Compute image-level Intersection over Union (IoU)
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

    # Print results
    print_results(auroc, mean_iou, avg_time)


if __name__ == "__main__":
    main()
