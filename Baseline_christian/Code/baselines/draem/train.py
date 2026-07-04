import torch, warnings
from anomalib.data import Folder
from anomalib.models import Draem
from anomalib.engine import Engine
from lightning.pytorch.callbacks import TQDMProgressBar
from baselines.utils import real_path
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Resize
from lightning.pytorch.loggers import CSVLogger


# -------------------
#    Configuration
# -------------------
warnings.filterwarnings("ignore", category = UserWarning)
INPUT_DIR = real_path("/dataset")
OUTPUT_DIR = real_path("/results")
IMG_SIZE = 256


# ----------------
#    Training
# ----------------
def main():
    # Optimize matrix multiplication for NVIDIA GPUs
    torch.set_float32_matmul_precision('high')

    # Load the custom dataset
    datamodule = Folder(name = "thesis_data", root = INPUT_DIR, normal_dir = "train/good", train_batch_size = 4,
                        eval_batch_size = 4, num_workers = 4)

    # Handle initial image transformations
    pre_processor = PreProcessor(transform = Resize(size = (IMG_SIZE, IMG_SIZE)))

    # Model initialization
    model = Draem(pre_processor = pre_processor)

    # Manage device placement, logging, and the training loop.
    engine = Engine(accelerator = "auto", devices = 1, max_epochs = 20, default_root_dir = OUTPUT_DIR,
                    callbacks = [TQDMProgressBar(refresh_rate = 1)], logger = CSVLogger(save_dir = OUTPUT_DIR))

    # Execute the training pipeline
    engine.fit(model = model, datamodule = datamodule)


if __name__ == "__main__":
    main()
