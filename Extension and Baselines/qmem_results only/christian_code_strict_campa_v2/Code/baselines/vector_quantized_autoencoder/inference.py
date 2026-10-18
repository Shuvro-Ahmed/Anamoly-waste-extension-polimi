import time
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from baselines.utils import *
from baselines.vector_quantized_autoencoder.architecture.vq_ae import VQVAE


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path('/baselines/vector_quantized_autoencoder/output')
CHECKPOINT_DIR = real_path('/baselines/vector_quantized_autoencoder/checkpoint')
IMG_SIZE = 256


# ---------------
#    Inference
# ---------------
def main():
    # Ensure directories exists
    ensure_dir([OUTPUT_DIR, CHECKPOINT_DIR])

    # Standardized preprocessing
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    # Initialize the model, move to the target device, and set in evaluation mode
    vqvae = VQVAE().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pth")
    vqvae.load_state_dict(torch.load(checkpoint_path, map_location = DEVICE, weights_only = True))
    vqvae.eval()

    # Containers for global evaluation metrics
    pixel_scores_all, pixel_labels_all, iou_list, inference_times = [], [], [], []

    # Retrieve the sorted list of image paths
    image_paths = load_test_images()

    # Process each image
    for img_path in tqdm(image_paths, desc = "Processing images"):
        # Load the corresponding ground truth mask
        mask_path, name = load_test_masks(img_path)

        start_time = time.time()

        # Load and resize image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to binary
        mask_np = (np.array(mask.resize((IMG_SIZE, IMG_SIZE))) > 127).astype(np.uint8)

        # Prepare input tensor for the model
        x_vis = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            recon, _ = vqvae(img_tensor)

        # Convert reconstruction back to numpy format
        recon_np = (recon.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)
        x_np = x_vis.astype(np.float32)

        # Compute pixel-wise Mean Absolute Error across channels and apply Gaussian smoothing
        err_map = np.mean(np.abs(x_np - recon_np), axis = 2)
        err_map = gaussian_filter(err_map, sigma = 1)

        # Threshold error map using the 90th percentile
        thr = np.percentile(err_map, 90)
        predicted_mask = (err_map >= thr).astype(np.uint8)

        # Remove small connected components (denoising)
        labeled_mask, num_features = label(predicted_mask)
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < 100:
                predicted_mask[labeled_mask == i] = 0

        # Record inference time
        inference_times.append(time.time() - start_time)

        # Accumulate metrics
        pixel_scores_all.append(err_map.flatten())
        pixel_labels_all.append(mask_np.flatten())

        # Compute IoU
        iou = compute_iou(predicted_mask, mask_np)
        iou_list.append(iou)

        # Save visualization of current image results
        visualize(x_vis, recon_np, predicted_mask, mask_np, iou, OUTPUT_DIR, name)

    # Final evaluation
    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)
    auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
    mean_iou = np.mean(iou_list).astype(float)
    avg_time = np.mean(inference_times).astype(float)

    # Print final results
    print_results(auroc, mean_iou, avg_time)


if __name__ == "__main__":
    main()
