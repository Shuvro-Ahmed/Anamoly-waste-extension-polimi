import subprocess, sys
from baselines.utils import *


# -------------------
#    Configuration
# -------------------
OUTPUT_DIR = real_path("/baselines/diffusion_model/checkpoint")


# ----------------
#    Training
# ----------------
def main():
    # Ensure directories exists
    ensure_dir([OUTPUT_DIR])

    # Training command
    command = [
        "accelerate", "launch",
        # Training script,
        "--train_data_dir", train_path(),
        "--output_dir", OUTPUT_DIR,
        "--resolution", "256",
        "--train_batch_size", "4",
        "--num_epochs", "30",
        "--learning_rate", "5e-5",
        "--lr_scheduler", "cosine",
        "--mixed_precision", "fp16",
        "--use_ema",
        "--gradient_accumulation_steps", "2"
    ]

    # Run training
    process = subprocess.Popen(command, stdout = sys.stdout, stderr = sys.stderr, shell = True)
    process.communicate()


if __name__ == "__main__":
    main()
