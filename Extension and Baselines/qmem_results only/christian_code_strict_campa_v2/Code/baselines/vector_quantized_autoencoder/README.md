# Vector-Quantized Autoencoder
We implement a _Vector-Quantized Variational Autoencoder_ (_VQ-VAE_) for anomaly detection. 
Unlike standard _VAEs_ that use a continuous latent space, the _VQ-VAE_ employs a discrete codebook to represent features.

By forcing the model to reconstruct images using a finite set of learned embedding vectors, the _VQ-VAE_ effectively filters out high-frequency noise and compresses the "
normal data distribution. 
Anomalies, which cannot be accurately represented by the discrete normal codebook entries, result in significantly higher reconstruction errors.

## Architecture details
The model consists of a convolutional pipeline with a discrete bottleneck, in particular: 

- _Encoder_: a three-layer convolutional network that downsamples the input image by a factor of 4 while increasing the depth to 256 channels, ending with a projection to the embedding dimension.

- _Bottleneck_: the codebook contains 1024 discrete embedding vectors and maps continuous encoder outputs to the nearest neighbor in the codebook using Euclidean distance.
A straight-through estimator allows gradients to flow back to the encoder despite the non-differentiable argmin operation.

- _Decoder_: a symmetric transposed convolutional network that upsamples the quantized latents back to the original image resolution (256×256).


## Installation
The implementation relies on standard PyTorch and Torchvision libraries. 
To set up the environment, run:

```bash
pip install torch torchvision tqdm scipy scikit-learn pillow
```

# Usage
To train the model on the custom _Seruso_ dataset, run:

```bash
python3 training.py
```

To evaluate the model using a saved checkpoint, run:

```bash
python3 inference.py
```

# Results
The performance of _VQ_VAE_ on the _Seruso_ dataset is summarized below. 

<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| VQ-VAE          | 82.96     | 19.52   | 0.06               | 16.7              |
</div>