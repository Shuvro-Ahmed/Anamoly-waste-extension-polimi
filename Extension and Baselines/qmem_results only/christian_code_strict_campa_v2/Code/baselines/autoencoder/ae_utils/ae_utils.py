import cv2
from PIL import Image
from torch.utils.data import Dataset

from baselines.utils import *


class RandomRectMask:
    """
    Generates random rectangular masks for image inpainting tasks.

    This class creates binary masks where '1' represents a hole and '0' represents the valid image area.
    The number, size, and position of these holes are randomized within specified constraints.

    Attributes:
        min_frac (float): Minimum dimension of a hole as a fraction of image size.
        max_frac (float): Maximum dimension of a hole as a fraction of image size.
        num_holes (tuple[int, int]): A range (min, max) for the number of holes per mask.
    """

    def __init__(self, min_frac = 0.05, max_frac = 0.25, num_holes = (1, 3)):
        """
        Initializes the mask generator with geometric constraints.

        Args:
            min_frac (float): Minimum fraction of the image height/width for a rectangle.
            max_frac (float): Maximum fraction of the image height/width for a rectangle.
            num_holes (tuple): Range (min, max) for the number of rectangles per mask.
        """

        self.min_frac = min_frac
        self.max_frac = max_frac
        self.num_holes = num_holes

    def __call__(self, height: int, width: int):
        """
        Generates a random rectangular mask based on input dimensions.

        Args:
            height (int): Height of the output mask in pixels.
            width (int): Width of the output mask in pixels.

        Returns:
            mask (np.ndarray): A 2D array of shape with the image size representing the generated mask.
        """

        # Initialize a clear mask (all zeros)
        mask = np.zeros((height, width), dtype = np.float32)

        # Determine the number of holes for this specific instance
        n = np.random.randint(self.num_holes[0], self.num_holes[1] + 1)

        for _ in range(n):
            # Sample random proportions for the hole's dimensions
            height_fraction, width_fraction = np.random.uniform(self.min_frac, self.max_frac, 2)

            # Convert relative fractions to absolute pixel values
            pixel_height, pixel_width = max(1, int(height * height_fraction)), max(1, int(width * width_fraction))

            # Sample the position
            y0 = np.random.randint(0, max(1, height - pixel_height))
            x0 = np.random.randint(0, max(1, width - pixel_width))

            # Apply the rectangle by setting the sliced area to 1
            mask[y0:y0 + pixel_height, x0:x0 + pixel_width] = 1.0

        return mask


class RandomBrushMask:
    """
    Generates random brush-like masks for image inpainting.

    This class simulates hand-drawn brush strokes.
    Instead of straight lines, it iteratively places circles that slightly shift in position and thickness.

    Attributes:
        min_strokes (int): Minimum number of independent strokes per mask.
        max_strokes (int): Maximum number of independent strokes per mask.
        min_len (int): Minimum number of iterations for a single stroke.
        max_len (int): Maximum number of iterations for a single stroke.
        min_w (int): Minimum radius (thickness) of the brush.
        max_w (int): Maximum radius (thickness) of the brush.
    """

    def __init__(self, min_strokes = 1, max_strokes = 3, min_len = 10, max_len = 100, min_w = 10, max_w = 60):
        """
        Initializes the mask generator with stroke and brush constraints.

        Args:
            min_strokes (int): The lower bound of strokes to generate.
            max_strokes (int): The upper bound of strokes to generate.
            min_len (int): Minimum points to draw per stroke path.
            max_len (int): Maximum points to draw per stroke path.
            min_w (int): Minimum brush thickness in pixels.
            max_w (int): Maximum brush thickness in pixels.
        """

        self.min_strokes = min_strokes
        self.max_strokes = max_strokes
        self.min_len = min_len
        self.max_len = max_len
        self.min_w = min_w
        self.max_w = max_w

    def __call__(self, height: int, width: int):
        """
        Generates a random brush mask for a given image size.

        Args:
            height (int): Height of the resulting mask.
            width (int): Width of the resulting mask.

        Returns:
            mask (np.ndarray): A 2D array of shape with the image size representing the generated mask.
        """

        # Initialize a clear mask (all zeros)
        mask = np.zeros((height, width), dtype = np.float32)

        # Determine the number of brushes for this specific instance
        num_strokes = np.random.randint(self.min_strokes, self.max_strokes + 1)

        for _ in range(num_strokes):
            # Select a random starting coordinate for the brush tip
            x, y = np.random.randint(0, width), np.random.randint(0, height)

            # Determine the number of iterations for this stroke
            length = np.random.randint(self.min_len, self.max_len)

            for _ in range(length):
                # Randomize thickness at each step to simulate varying brush pressure
                thickness = np.random.randint(self.min_w, self.max_w)

                # Move the brush tip by a random displacement
                x = np.clip(x + np.random.randint(-20, 21), 0, width - 1)
                y = np.clip(y + np.random.randint(-20, 21), 0, height - 1)

                # Draw a filled circle at the new coordinate.
                cv2.circle(mask, (int(x), int(y)), int(thickness), 1, -1)

        return mask


class DatasetLoaderWithProcessing(Dataset):
    """PyTorch dataset loader with integrated preprocessing and mask generation.

        This dataset handles the loading of images from a directory, applies random masking, and formats the data.

        Attributes:
            paths (list[Path]): A sorted list of all valid image file paths found.
            img_size (int): The target height and width for the square resizing.
            rect (RandomRectMask): Instance of the rectangular mask generator.
            brush (RandomBrushMask): Instance of the brush-stroke mask generator.
            use_brush_prob (float): Probability of selecting a brush mask over a rectangle.
        """

    def __init__(self, data_dir: str, img_size: int):
        """
        Initializes the loader and scans for image files.

        Args:
            data_dir (str): Path to the directory containing images.
            img_size (int): Dimensions to which images are resized (square).
        """

        self.paths = [p for p in sorted(Path(data_dir).rglob("*")) if is_image_file(p)]
        self.img_size = img_size
        self.rect = RandomRectMask()
        self.brush = RandomBrushMask()
        self.use_brush_prob = 0.5

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """

        return len(self.paths)

    def __getitem__(self, idx):
        """Loads and processes a single sample from the dataset.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            element (dict): A dictionary containing:
                - image (torch.Tensor): 4-channel input (RGB + Mask).
                - target (torch.Tensor): 3-channel original RGB image.
                - path (str): The absolute file path of the image.
        """

        # Load the image at the given index and convert to a numpy array
        p = self.paths[idx]
        img = np.array(Image.open(p).convert("RGB"))

        # Resize image to self.img_size x self.img_size
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)
        h, w = img.shape[:2]

        # Generate mask: choose the brush or rectangle randomly
        mask = self.brush(h, w) if np.random.rand() < self.use_brush_prob else self.rect(h, w)

        # Resize mask to the same size
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation = cv2.INTER_NEAREST)
        mask = mask[..., None]

        # Normalize image to [0, 1]
        img_f = img.astype(np.float32) / 255.0

        # Apply mask to image
        masked_img = img_f * (1.0 - mask)

        # Concatenate masked image and mask as input
        inp = np.concatenate([masked_img, mask], axis = 2).transpose(2, 0, 1)
        target = img_f.transpose(2, 0, 1)

        return {'image': torch.from_numpy(inp), 'target': torch.from_numpy(target), 'path': str(p)}
