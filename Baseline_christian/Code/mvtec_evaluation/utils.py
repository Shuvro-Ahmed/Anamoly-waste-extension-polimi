import glob, torch, os, re
from pathlib import Path
from typing import List, Tuple, Union
import torch.nn.functional as funct
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import jaccard_score
from torchvision import transforms
from tqdm import tqdm
from proposed_solution.architecture.model import VQVAE2


# -------------------
#    Configuration
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------
#    Utils methods
# -------------------
def reassemble_crops_to_full(crops, full_size):
    """
    Reassembles a batch of four corner crops per image into full-sized reconstructed images.

    Each crop is first upsampled to match the target spatial resolution, then placed in its corresponding quadrant.

    Args:
        crops (torch.Tensor): tensor of shape (4 * B, C, crop_size, crop_size) containing the four crops for each of
                              the B images.
        full_size (int): target height/width of the reconstructed full image.

    Returns:
        reconstruction (torch.Tensor): a tensor of shape (B, C, full_size, full_size) containing the reassembled images.
    """

    # Extract the batch size and crop size from the crops tensor
    batch, channels, crop_size, _ = crops.shape
    batch = batch // 4

    # Initialize the output tensor with zeros
    recons = torch.zeros((batch, channels, full_size, full_size), device = crops.device)

    # Upsample the crops and place them in their corresponding quadrants
    scale_factor = full_size / crop_size
    crops_up = funct.interpolate(crops, size = (int(crop_size * scale_factor), int(crop_size * scale_factor)),
                                 mode = 'bilinear', align_corners = False)
    end = int(crop_size * scale_factor)

    # Iterate over each image in the batch and place the four crops in their corresponding quadrants
    for i in range(batch):
        recons[i, :, 0:end, 0:end] = crops_up[i * 4 + 0]
        recons[i, :, 0:end, full_size - end:full_size] = crops_up[i * 4 + 1]
        recons[i, :, full_size - end:full_size, 0:end] = crops_up[i * 4 + 2]
        recons[i, :, full_size - end:full_size, full_size - end:full_size] = crops_up[i * 4 + 3]

    return recons


def get_corner_crops(img, crop_size):
    """
    Extracts the four corner crops from an image tensor.

    Args:
        img (torch.Tensor): a 3D image tensor of shape (C, H, W).
        crop_size (int): the height and width of each square crop.

    Returns:
        crops (torch.Tensor): a tensor of shape (4, C, crop_size, crop_size) containing the four corner crops.
    """

    # Extract the image dimensions
    _, height, width = img.shape

    # Extract the top-left, top-right, bottom-left, and bottom-right crops from the image
    crops = torch.stack([img[:, 0:crop_size, 0:crop_size],
                         img[:, 0:crop_size, width - crop_size:width],
                         img[:, height - crop_size:height, 0:crop_size],
                         img[:, height - crop_size:height, width - crop_size:width]])

    return crops


def real_path(path: str) -> str:
    """
    Return the absolute path for a given relative path inside the project.

   Args:
       path (str): Relative path from the project root (two levels up).

   Returns:
       output (str): Absolute path corresponding to the given relative path.
   """

    return str(os.path.realpath(f"../{path}"))


def ensure_dir(paths: Union[List[Path], List[str]]):
    """
    Creates directories if they do not exist.

    This function creates all directories in the given list of paths,
    including any necessary parent directories.

    Args:
        paths (List[Path] or List[str]): List of directory paths to create.
    """

    for p in paths:
        path_obj = Path(p)
        path_obj.mkdir(parents = True, exist_ok = True)


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between a predicted mask and ground truth mask.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        true_mask (np.ndarray): Ground truth binary mask.

    Returns:
        iou (float): IoU score between 0 and 1. Returns 1.0 if both masks are empty.
    """

    # Both masks are empty, define IoU as 1.0
    if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0:
        return 1.0

    # Compute Jaccard score
    return jaccard_score(true_mask.flatten(), pred_mask.flatten())


def natural_key(path: Union[str, Path]) -> List[Union[str, int]]:
    """
    Generate a natural sorting key for file paths.

    Args:
        path (str or Path): File path to generate the sorting key for.

    Returns:
        output (List[Union[str, int]]): List containing strings and integers extracted from the filename.
    """

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(path))]


def load_test_images(path: str) -> List[str]:
    """
    Loads and returns a sorted list of test image file paths from a category.

    This function traverses the test directory structure, which contains subdirectories for different defect types.
    It gathers all image files and sorts them naturally to ensure consistent ordering.

    Args:
        path (str): The system path to the category test directory.

    Returns:
        List[str]: A list of absolute file paths to the discovered images, sorted alphabetically.
    """

    # Convert to the absolute path to avoid issues with relative pathing in different OS environments
    path = os.path.realpath(path)

    # Load the image files
    extensions = ['*.png', '*.jpg', '*.JPEG', '*.bmp']
    image_files = []
    for ext in extensions: image_files.extend(glob.glob(os.path.join(path, "**", ext), recursive = True))

    # Sort files naturally
    return sorted(image_files, key = natural_key)


def load_test_masks(img_path: str, category_test_path: str) -> Tuple[str, str]:
    """
    Resolves the ground truth mask path and generates a unique identifier for an image.

    This function maps a test image to its corresponding binary mask by traversing the directory structure.
    It also creates a unique name used for saving visualization results, combining the defect type and the filename.

    Args:
        img_path (str): The absolute path to the specific test image.
        category_test_path (str): The path to the test directory of the current category.

    Returns:
        Tuple[Optional[str], str]: A tuple containing:
            - mask_path: The absolute path to the mask image, or None if the image is from the normal class.
            - unique_name: A string identifier used for output files.
    """

    # Determine the defect type and filename
    rel_path = os.path.relpath(img_path, category_test_path)
    defect_type, filename = os.path.split(rel_path)

    # Generate a unique name for visualization outputs
    unique_name = f"{defect_type}_{os.path.splitext(filename)[0]}"

    # Handle ground truth path resolution
    category_root = os.path.dirname(os.path.realpath(category_test_path))
    if defect_type == "good":
        mask_path = None
    else:
        mask_filename = os.path.splitext(filename)[0] + "_mask.png"
        mask_path = os.path.join(category_root, "ground_truth", defect_type, mask_filename)

    return mask_path, unique_name


def visualize(original_image: np.ndarray, reconstructed_image: np.ndarray, predicted_mask: np.ndarray,
              ground_truth_mask: np.ndarray, iou: float, output_dir: str, name: str) -> None:
    """
    Visualize original, reconstructed images and predicted and ground truth masks.

    This function creates a 1x4 subplot showing the original image, the reconstructed image, the predicted mask,
    and the ground truth mask.
    The figure is saved to the specified output directory with the given filename.
    The IoU score is displayed as the figure title.

    Args:
        original_image (np.ndarray): Original input image.
        reconstructed_image (np.ndarray): Reconstructed image (predicted output).
        predicted_mask (np.ndarray): Predicted mask from the model.
        ground_truth_mask (np.ndarray): Ground truth binary mask.
        iou (float): Intersection over Union score between predicted and ground truth masks.
        output_dir (str): Directory to save the visualization.
        name (str): Filename (with extension) to save the figure as.
    """

    # Create a 1x4 figure for original, reconstructed, predicted mask, and ground truth mask
    fig, axs = plt.subplots(1, 4, figsize = (14, 4))

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].set_title("Original")

    # Display the reconstructed image (ensure type is uint8)
    axs[1].imshow(reconstructed_image)
    axs[1].set_title("Reconstruction")

    # Display the predicted mask in grayscale
    axs[2].imshow(predicted_mask, cmap = "gray")
    axs[2].set_title("Predicted Mask")

    # Display the ground truth mask in grayscale
    axs[3].imshow(ground_truth_mask, cmap = "gray")
    axs[3].set_title("Ground Truth Mask")

    # Remove axes for all subplots
    for a in axs: a.axis("off")

    # Add IoU as the figure title
    plt.suptitle(f"IoU = {iou:.3f}", fontsize = 12, fontweight = "bold")
    plt.tight_layout()

    # Save the figure to the output directory
    output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.png")
    plt.savefig(output_path, dpi = 150)
    plt.close(fig)


def print_results(auroc: float, mean_iou: float, avg_time: float) -> None:
    """
    Print final evaluation metrics in a formatted manner.

    Args:
        auroc (float): Pixel-wise Area Under the Receiver Operating Characteristic.
        mean_iou (float): Mean Intersection over Union score.
        avg_time (float): Average inference time in seconds.
    """

    print("\n" + "=" * 30)
    print(f"Pixel AUROC           : {auroc:.4f}")
    print(f"Mean IoU              : {mean_iou:.4f}")
    print(f"Avg Inference Time (s): {avg_time:.4f}")
    print("=" * 30 + "\n")


def compute_statistics(stats_path: str, img_size: int, k: float, category_train_dir: str,
                       checkpoint_path: str) -> float:
    """
    Compute or load the error distribution statistics for the training set.

    This function iterates through the normal training images, calculates the reconstruction error using the VQ-VAE2
    model, and determines the 3rd quartile (Q3) and Interquartile Range (IQR).

    Args:
        stats_path (str): Path to the .npz file where statistics are or will be stored.
        img_size (int): The target image resolution for resizing.
        k (float): The multiplier for the IQR to determine the sensitivity of the threshold.
        category_train_dir (str): Path to the directory containing the training images for the current category.
        checkpoint_path (str): Path to the checkpoint file for the VQ-VAE2 model.

    Returns:
        threshold (float): The calculated anomaly detection threshold.
    """

    # Check if statistics were already computed
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return float(stats["q3"]) + k * float(stats["iqr"])

    os.makedirs(os.path.dirname(stats_path), exist_ok = True)

    # Set up the specific model for this category
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    autoencoder = VQVAE2(num_embeddings = 256).to(DEVICE)
    autoencoder.load_state_dict(torch.load(checkpoint_path, map_location = DEVICE, weights_only = True))
    autoencoder.eval()

    # 3. Retrieve good images for this specific category
    search_path = os.path.join(category_train_dir, "good", "*")
    image_paths = sorted(glob.glob(search_path), key = natural_key)

    all_errors = []

    with torch.no_grad():
        for img_path in tqdm(image_paths,
                             desc = f"Stats calculation [{os.path.basename(os.path.dirname(category_train_dir))}]"):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            recon, _ = autoencoder(x)

            x_np = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            recon_np = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # Error map calculation with distance and Gaussian smoothing
            err_map = np.mean(np.abs(x_np - recon_np), axis = 2)
            err_map = gaussian_filter(err_map, sigma = 1)

            all_errors.append(err_map.flatten())

    # Compute distribution stats
    all_errors = np.concatenate(all_errors)
    q1 = np.percentile(all_errors, 25).astype(float)
    q3 = np.percentile(all_errors, 75).astype(float)
    iqr = q3 - q1

    stats = {"q3": float(q3), "iqr": float(iqr)}
    np.savez(stats_path, **stats)

    return float(stats["q3"]) + k * float(stats["iqr"])
