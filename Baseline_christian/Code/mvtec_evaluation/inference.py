from utils import *
import time, torch
import numpy as np
from proposed_solution.architecture.model import VQVAE2
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from sklearn.metrics import roc_auc_score
from torchvision import transforms


# -------------------
#    Configuration
# -------------------
IMG_SIZE = 512
DATASET_ROOT = real_path("dataset_mvtec")
CHECKPOINT_BASE = "checkpoint"
OUTPUT_BASE = "predictions"
STATS_DIR = "statistics"


# ---------------
#    Inference
# ---------------
def main(category_name):
    # Set up directories and paths
    category_test_dir = os.path.join(DATASET_ROOT, category_name, "test")
    ckp_path = os.path.join(CHECKPOINT_BASE, category_name, "model.pth")
    output_dir = os.path.join(OUTPUT_BASE, category_name)
    stats_path = os.path.join(STATS_DIR, f"{category_name}_stats.npz")
    ensure_dir([output_dir, STATS_DIR])

    # Load or compute the threshold used to separate normal from anomalous pixels
    train_path = os.path.join(DATASET_ROOT, category_name, "train")
    thr = compute_statistics(stats_path, IMG_SIZE, 4, category_train_dir = train_path, checkpoint_path = ckp_path)

    # Model initialization with a smaller embedding size
    autoencoder = VQVAE2(num_embeddings = 256).to(DEVICE)
    autoencoder.load_state_dict(torch.load(ckp_path, map_location = DEVICE, weights_only = True))
    autoencoder.eval()

    # Preprocess the data
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    image_paths = load_test_images(category_test_dir)

    # Define connectivity for morphological operations
    struct = generate_binary_structure(2, 2)

    # Containers for global evaluation metrics
    iou_list, inference_times, auc_scores, auc_labels = [], [], [], []

    for img_path in tqdm(image_paths, desc = f"Inference [{category_name}]"):
        mask_path, name = load_test_masks(img_path, category_test_dir)

        start_time = time.time()

        # Load and format the image
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            recon, _ = autoencoder(x)

        # Pixel-wise error map and Gaussian smoothing
        x_np = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        recon_np = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Pixel-wise Mean Absolute Error across channels and apply Gaussian smoothing
        err_map = np.mean(np.abs(x_np - recon_np), axis = 2)
        err_map = gaussian_filter(err_map, sigma = 1)

        # Load or create the mask depending on the folder category
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask_np = (transform(mask).squeeze().numpy() > 0.5).astype(np.uint8)
        else:
            mask_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype = np.uint8)

        # Collect data for pixel-level AUROC calculation
        auc_scores.append(err_map.flatten())
        auc_labels.append(mask_np.flatten())

        # Morphological operations
        raw_pred = err_map > thr
        cleaned = binary_closing(raw_pred, structure = struct, iterations = 2)
        cleaned = binary_opening(cleaned, structure = struct, iterations = 1)
        cleaned = binary_fill_holes(cleaned)

        # Remove tiny clusters
        labeled, num_features = label(cleaned)
        pred_mask = np.zeros_like(cleaned, dtype = np.uint8)
        for i in range(1, num_features + 1):
            comp = (labeled == i)
            if np.sum(comp) >= 50:
                pred_mask[comp] = 1

        inference_times.append(time.time() - start_time)

        # Compute Intersection over Union (IoU) for the current image
        iou_list.append(compute_iou(pred_mask, mask_np))

        # Save result visualization
        visualize(x_np, recon_np, pred_mask, mask_np, iou_list[-1], output_dir, name)

    # Flatten the results across all images in the category
    auc_scores_cat = np.concatenate(auc_scores)
    auc_labels_cat = np.concatenate(auc_labels)

    # Compute the final metrics
    auroc = roc_auc_score(auc_labels_cat, auc_scores_cat) if len(np.unique(auc_labels_cat)) > 1 else np.nan
    iou = np.mean(iou_list).astype(float)
    inference_time = np.mean(inference_times).astype(float)

    # Print the final metrics
    print_results(auroc, iou, inference_time)


if __name__ == "__main__":
    # Detect all category folders
    categories = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]

    for category in categories:
        main(category)
