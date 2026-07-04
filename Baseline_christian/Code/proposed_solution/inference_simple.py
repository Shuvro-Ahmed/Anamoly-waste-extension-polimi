from utils import *
import time
from proposed_solution.architecture.model import VQVAE2
from transformers import Sam3Processor, Sam3Model
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from sklearn.metrics import roc_auc_score
from torchvision import transforms


# -------------------
#    Configuration
# -------------------
IMG_SIZE = 256
NO_WHITE = True
STATS_PATH = f"statistics/train_quartiles_{IMG_SIZE}.npz"
CHECKPOINT_DIR = "checkpoint/model.pth"
INPUT_DIR = test_path()


# ---------------
#    Inference
# ---------------
def main():
    # Prepare directories and load images
    if NO_WHITE:
        input_dir, output_dir = real_path("/dataset/test_cleaned"), "predictions/base_no_white"
    else:
        input_dir, output_dir = real_path("/dataset/test"), "predictions/base"
    ensure_dir([input_dir, output_dir])
    image_paths = load_test_images(input_dir)

    # Calculate the anomaly threshold
    thr = compute_statistics(STATS_PATH, IMG_SIZE, 4)

    # Prepare image transforms and load the model in evaluation mode
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    autoencoder = VQVAE2().to(DEVICE)
    autoencoder.load_state_dict(torch.load(CHECKPOINT_DIR, map_location = DEVICE, weights_only = True))
    autoencoder.eval()

    # Initialize tracking metrics
    iou_list, inference_times, auc_scores, auc_labels = [], [], [], []

    # Connectivity structure for morphology
    struct = generate_binary_structure(2, 2)

    for img_path in tqdm(image_paths, desc = "Processing images"):
        mask_path, name = load_test_masks(img_path, input_dir)

        start_time = time.time()

        # Preprocessing inputs and ground truth masks
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np = (transform(mask).squeeze().numpy() > 0.5).astype(np.uint8)

        # Identifies pixels masked in the original but not in the cleaned set
        suppression_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype = np.uint8)
        if NO_WHITE:
            orig_mask_path = os.path.join(real_path("/dataset/test/masks"), os.path.splitext(name)[0] + ".png")
            orig_mask = Image.open(orig_mask_path).convert("L")
            orig_m_np = (transform(orig_mask).squeeze().numpy() > 0.5).astype(np.uint8)
            suppression_mask = np.logical_and(orig_m_np == 1, mask_np == 0).astype(np.uint8)

        # Forward pass
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon, _ = autoencoder(x)

        # Convert tensors to numpy arrays for pixel-wise error calculation
        x_np = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        recon_np = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Compute pixel-wise error and smooth to reduce high-frequency noise
        err_map = np.mean(np.abs(x_np - recon_np), axis = 2)
        err_map = gaussian_filter(err_map, sigma = 1)

        # Flatten arrays
        flat_scores = err_map.flatten()
        flat_labels = mask_np.flatten()

        if NO_WHITE:
            # Exclude suppressed indices from AUROC calculation to avoid bias
            valid_indices = (suppression_mask.flatten() == 0)
            auc_scores.append(flat_scores[valid_indices])
            auc_labels.append(flat_labels[valid_indices])
        else:
            auc_scores.append(flat_scores)
            auc_labels.append(flat_labels)

        # Segment the anomaly based on the training distribution threshold
        raw_pred = err_map > thr

        # Closing fills small dark holes and opening removes small bright noise
        cleaned = binary_closing(raw_pred, structure = struct, iterations = 2)
        cleaned = binary_opening(cleaned, structure = struct, iterations = 1)
        cleaned = binary_fill_holes(cleaned)

        # Remove tiny noise clusters
        labeled, num_features = label(cleaned)
        pred_mask = np.zeros_like(cleaned, dtype = np.uint8)
        for i in range(1, num_features + 1):
            comp = (labeled == i)
            if np.sum(comp) >= 50:
                pred_mask[comp] = 1

        # Ensure suppressed regions are not counted as predicted anomalies
        if NO_WHITE:
            pred_mask[suppression_mask == 1] = 0

            # Opening removes the small bright noise
            pred_mask = binary_opening(pred_mask, structure = struct, iterations = 2)

        # Performance tracking
        inference_times.append(time.time() - start_time)
        iou = compute_iou(pred_mask, mask_np)
        iou_list.append(iou)

        # Save visualization
        visualize(x_np, recon_np, pred_mask, mask_np, iou, output_dir, name)

    # Final statistics aggregation
    auc_scores_cat = np.concatenate(auc_scores)
    auc_labels_cat = np.concatenate(auc_labels)

    # Final statistics computation
    auroc = roc_auc_score(auc_labels_cat, auc_scores_cat)
    mean_iou = np.mean(iou_list).astype(float)
    avg_time = np.mean(inference_times).astype(float)

    # Print results
    print_results(auroc, mean_iou, avg_time)


if __name__ == "__main__":
    main()
