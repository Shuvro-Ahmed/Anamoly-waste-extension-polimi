# Diffusion Model

We implement a _Denoising Diffusion Probabilistic Model_ (DDPM) to detect anomalies via iterative reconstruction. 
Unlike a standard _Autoencoder_ that performs a single-pass compression, the _Diffusion Model_ learns to reverse a multi-step Gaussian noise process to recover normal data distributions.

Anomalies are identified by comparing the original image with its denoised reconstruction. 
Because the model is trained exclusively on normal samples, it fails to accurately reconstruct anomalous textures, resulting in high reconstruction errors in those specific regions.

## Architecture details
The model utilizes the _U-Net_ architecture from the Hugging Face diffusers library, paired with a linear DDPMScheduler.
The model starts from pure Gaussian noise and iteratively refines the image over 50 inference timesteps to generate a clean version of the input.
The network operates on RGB images resized to 256×256. 

For the post-processing we apply a Gaussian filter to the error map to reduce pixel noise, and then we generate the anomaly map using a 90th percentile threshold.

## Installation 
To set up the environment, clone the official diffusers repository and install the library along with its training dependencies:

```bash
pip install diffusers accelerate transformers
```

> _Citation_: von Platen, P., Patil, S., Lozhkov, A., et al. (2022). Diffusers: State-of-the-art diffusion models. GitHub. https://github.com/huggingface/diffusers

## Usage 
To train the model using accelerate, run:

```bash
python3 training.py
```

To perform inference and generate anomaly maps using the DDPM reverse process, run:

```bash
python3 inference.py
```

## Results
The performance of _Diffusion Model_ on the _Seruso_ dataset is summarized below. 
<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| Diffusion Model     | 87.16     | 34.84   | 1.89               | 0.53              |
			
</div>