import time, warnings
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from anomalib.models import Draem
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
warnings.filterwarnings("ignore", category = UserWarning)
OUTPUT_DIR = real_path('/baselines/draem/output')
CHECKPOINT_PATH = real_path('/baselines/draem/checkpoint/model.ckpt')
IMG_SIZE = 256


# ---------------
#    Inference
# ---------------
def main():
    # Ensure all required directories exist
    ensure_dir([OUTPUT_DIR])

    # Load pretrained model
    model = Draem.load_from_checkpoint(CHECKPOINT_PATH, weights_only = False)
    model.to(DEVICE)
    model.eval()

    # Evaluation containers
    pixel_scores_all, pixel_labels_all, iou_list, inference_times = [], [], [], []

    # Get sorted test images
    image_paths = load_test_images()

    # Image processing loop
    for img_path in tqdm(image_paths, desc = "Processing images"):
        mask_path, name = load_test_masks(img_path)

        start_time = time.perf_counter()

        # Image and mask preprocessing
        img_pil = Image.open(img_path).convert("RGB")
        x_vis = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)

        img_tensor = torch.from_numpy(x_vis.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        mask_pil = Image.open(mask_path).convert("L")
        mask_np = (np.array(mask_pil.resize((IMG_SIZE, IMG_SIZE))) > 127).astype(np.uint8)

        # Model forward pass
        with torch.no_grad():
            output = model(img_tensor)

        # Heatmap and mask computation
        err_map = output.anomaly_map[0].squeeze().cpu().numpy()
        predicted_mask = output.pred_mask[0].squeeze().cpu().numpy().astype(np.uint8)

        # Apply smoothing and remove small noise components
        err_map = gaussian_filter(err_map, sigma = 1)
        labeled_mask, num_features = label(predicted_mask)
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < 100:
                predicted_mask[labeled_mask == i] = 0

        inference_times.append(time.perf_counter() - start_time)

        # Record metrics
        pixel_scores_all.append(err_map.flatten())
        pixel_labels_all.append(mask_np.flatten())
        iou = compute_iou(predicted_mask, mask_np)
        iou_list.append(iou)

        # Cast the results back to numpy
        recon_np = x_vis.astype(np.float32)

        # Save the results
        visualize(x_vis, recon_np, predicted_mask, mask_np, iou, OUTPUT_DIR, name)

    # Final result calculation
    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)
    auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
    mean_iou = np.mean(iou_list).astype(float)
    avg_time = np.mean(inference_times).astype(float)

    print_results(auroc, mean_iou, avg_time)


if __name__ == "__main__":
    main()
