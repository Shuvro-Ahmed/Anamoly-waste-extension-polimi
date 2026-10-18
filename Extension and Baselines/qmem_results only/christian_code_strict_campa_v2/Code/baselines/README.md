# Multi-Object Unsupervised Anomaly Detection Baselines

This folder contains all baseline methods used to benchmark modern anomaly detection approaches on the _Seruso_ dataset. 
These baselines serve as reference implementations for evaluating detection accuracy, localization quality, and computational efficiency.

## Features
Each baseline method is organized in its own folder and includes the following components:
- 📝 _Training script_: Python script used to train the model. The training procedure may vary depending on the specific model and library.
- 📝 _Inference script_: Python script used to evaluate the trained model. Anomalies are detected using a 90th-percentile reconstruction-error threshold. The script reports: quantitative metrics (AUROC, IoU, and inference time), and qualitative outputs (reconstructed images and predicted masks for every test image).
- 📂 _Output folder_: stores the full set of test images along with reconstructed images, predicted anomaly masks, and ground-truth comparisons
- 📂 _Checkpoint folder_: contains the saved model weights corresponding to the best training checkpoint.
- 📂 _Architecture folder_: includes the model definition used during training and inference (when a custom architecture is used).

Each model folder also provides detailed instructions for running training and inference.

## Results
The table below summarizes the quantitative performance of all evaluated methods on the _Seruso_ dataset.
For each method, we report:
- Detection performance: AUROC.
- Localization quality: IoU.
- Efficiency metrics: inference time and throughput.

This enables direct comparison across reconstruction-based, diffusion-based, and _DRÆM_ approaches.

<div align="center">
  
| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| Autoencoder     | 81.04     | 17.46   | 0.06               | 16.7              |
| VQ-VAE          | 82.96     | 19.52   | 0.06               | 16.7              |
| VQ-VAE2         | 83.14     | 19.35   | *0.05*               | *20.0*              |
| Diffusion Model | *87.16*     | *34.84*   | 1.89               | 0.53              |
| DRÆM            | 43.56     | 7.90      | 0.11                | 9.09                |

</div>
