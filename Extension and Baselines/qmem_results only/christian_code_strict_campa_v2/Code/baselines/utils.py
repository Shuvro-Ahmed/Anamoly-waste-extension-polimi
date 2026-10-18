import glob, os, torch, re
from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
from torch.nn import functional


# -------------------
#    Configuration
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------
#    Utils methods
# -------------------
def is_image_file(path: Path) -> bool:
    """
    Check if a file is an image based on its file extension.

    Args:
        path (Path): Path to the file to check.

    Returns:
        output (bool): True if the file has an image extension.
    """

    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def train_path() -> str:
    """
    Return the absolute path to the training dataset directory.

    Returns:
        output (str): Absolute path to the training dataset directory.
    """

    return str(os.path.realpath("../../dataset/train"))


def test_path() -> str:
    """
    Return the absolute path to the test dataset directory.

    Returns:
        output (str): Absolute path to the test dataset directory.
    """

    return str(os.path.realpath("../../dataset/test"))


def real_path(path: str) -> str:
    """
    Return the absolute path for a given relative path inside the project.

   Args:
       path (str): Relative path from the project root (two levels up).

   Returns:
       output (str): Absolute path corresponding to the given relative path.
   """

    return str(os.path.realpath(f"../../{path}"))


def ensure_dir(paths: Union[List[Path], List[str]]) -> None:
    """
    Create directories if they do not exist.

    This function iterates over a list of paths and ensures that each directory exists.
    If a directory does not exist, it is created, including all necessary parent directories.

    Args:
        paths (List[Path] or List[str]): List of directory paths to create.
    """

    for p in paths:
        Path(p).mkdir(parents = True, exist_ok = True)


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute the Intersection over Union between a predicted mask and the ground truth mask.

    The IoU, also called the Jaccard index, is computed as the ratio of the intersection over the union of the
    predicted and true binary masks.
    If both masks are empty, the function returns 1.0.

    Args:
        pred_mask (np.ndarray): Predicted mask.
        true_mask (np.ndarray): Ground truth binary mask.
        threshold (float, optional): Threshold to binarize the predicted mask. Defaults to 0.5.

    Returns:
        iou (float): IoU score between 0 and 1.
    """

    # Binarize the predicted mask using the given threshold
    pred_bin = (pred_mask > threshold).astype(np.uint8)

    # Ensure the ground truth mask is binary
    true_bin = (true_mask > 0.5).astype(np.uint8)

    iou = jaccard_score(true_bin.flatten(), pred_bin.flatten())

    return iou


def natural_key(path: Union[str, Path]) -> List[Union[str, int]]:
    """
    Generate a natural sorting key for file paths.

    Args:
        path (str or Path): File path to generate the sorting key for.

    Returns:
        output (List[Union[str, int]]): List containing strings and integers extracted from the filename.
    """

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(path))]


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight_hole: float = 1.0,
                               weight_context: float = 0.1) -> torch.Tensor:
    """
    Compute a masked reconstruction loss for image inpainting.

    The loss is a weighted sum of L1 losses computed separately inside the hole region and in the context.

    Args:
        pred (torch.Tensor): Predicted image tensor of shape (B, C, H, W).
        target (torch.Tensor): Ground truth image tensor of the same shape as `pred`.
        mask (torch.Tensor): Binary mask tensor of shape (B, 1, H, W) or (B, C, H, W).
        weight_hole (float, optional): Weight for the loss inside the hole. Defaults to 1.0.
        weight_context (float, optional): Weight for the loss in the context region. Defaults to 0.1.

    Returns:
        loss (torch.Tensor): Scalar tensor representing the weighted reconstruction loss.
    """

    # Compute L1 loss inside the hole region
    l1_hole = functional.l1_loss(pred * mask, target * mask)

    # Compute L1 loss in the context region
    l1_context = functional.l1_loss(pred * (1 - mask), target * (1 - mask))

    # Combine the two losses with specified weights
    loss = weight_hole * l1_hole + weight_context * l1_context

    return loss


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
    axs[1].imshow(reconstructed_image.astype(np.uint8))
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


def load_test_images() -> List[str]:
    """
    Load and return a sorted list of test image file paths.

    The function looks for images in the test dataset and sorts them using natural order.

    Returns:
        output (List[str]): Sorted list of image file paths.
    """

    # Resolve the absolute path to the test images directory
    path = os.path.realpath("../../dataset/test")

    # Use glob to find all files in the 'images' subdirectory
    image_files = glob.glob(os.path.join(path, "images", "*"))

    # Sort files naturally (handles numeric order in filenames)
    return sorted(image_files, key = natural_key)


def load_test_masks(img_path: str) -> Tuple[str, str]:
    """
    Return the corresponding mask path and image filename for a test image.

    Given the path to a test image, this function computes the corresponding mask file in the test dataset and also
    returns the image's filename.

    Args:
        img_path (str): Path to the test image.

    Returns:
        output (Tuple[str, str]): A tuple containing:
            - mask_path (str): Absolute path to the corresponding mask image.
            - name (str): Filename of the original test image.
    """

    # Extract the filename from the image path
    name = os.path.basename(img_path)

    # Resolve the absolute path to the test dataset
    path = os.path.realpath("../../dataset/test")

    # Construct the full path to the corresponding mask
    mask_path = os.path.join(path, "masks", os.path.splitext(name)[0] + ".png")

    return mask_path, name
