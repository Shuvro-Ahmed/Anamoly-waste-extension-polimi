from utils import *
import time
from transformers import Sam3Processor, Sam3Model
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from sklearn.metrics import roc_auc_score


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
        input_dir, output_dir = real_path("/dataset/test_cleaned"), "predictions/sam_no_white"
    else:
        input_dir, output_dir = real_path("/dataset/test"), "predictions/sam"
    ensure_dir([input_dir, output_dir])
    image_paths = load_test_images(input_dir)

    # Calculate the anomaly threshold
    thr = compute_statistics(STATS_PATH, IMG_SIZE, 4)

    # Prepare image transforms and load the model in evaluation mode
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    autoencoder = VQVAE2().to(DEVICE)
    autoencoder.load_state_dict(torch.load(CHECKPOINT_DIR, map_location = DEVICE, weights_only = True))
    autoencoder.eval()

    # Load SAM3 model and processor
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

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

        # Prompt SAM to find distinct items to generate a set of candidate object masks
        inputs_sam = sam3_processor(images = img, text = "distinct item", return_tensors = "pt").to(DEVICE)

        with torch.no_grad():
            sam_outputs = sam3_model(**inputs_sam)

        # Post-process the raw logits into binary masks based on specified thresholds
        sam_results = sam3_processor.post_process_instance_segmentation(
            sam_outputs,
            threshold = 0.5,
            mask_threshold = 0.5,
            target_sizes = inputs_sam["original_sizes"].tolist()
        )[0]

        # Define a variable that will store SAM segments that overlap significantly with VQ-VAE anomalies
        refined_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype = np.uint8)

        for m in sam_results["masks"].cpu().numpy():
            # Resize SAM's output mask to match the VQ-VAE internal image size
            m = Image.fromarray(m.astype(np.uint8))
            sm_resized = np.array(m.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST))

            # If more than 20% of a SAM object is covered by the anomaly prediction, we consider the object anomalous
            if (np.logical_and(sm_resized, pred_mask).sum() / max(sm_resized.sum(), 1)) >= 0.2:
                refined_mask[sm_resized > 0] = 1

        # Close internal gaps within the refined segments
        refined_mask = binary_closing(refined_mask, structure = struct, iterations = 1).astype(np.uint8)

        # Remove tiny residual artifacts that survived the refinement
        refined_labeled, ref_num = label(refined_mask)
        for i in range(1, ref_num + 1):
            if np.sum(refined_labeled == i) < 20:
                refined_mask[refined_labeled == i] = 0

        # Ensure suppressed regions are not counted as predicted anomalies
        if NO_WHITE:
            refined_mask[suppression_mask == 1] = 0

            # Opening removes the small bright noise
            refined_mask = binary_opening(refined_mask, structure = struct, iterations = 1)

        # Performance tracking
        inference_times.append(time.time() - start_time)
        iou = compute_iou(refined_mask, mask_np)
        iou_list.append(iou)

        # Save visualization
        visualize(x_np, recon_np, refined_mask, mask_np, iou, output_dir, name)

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
