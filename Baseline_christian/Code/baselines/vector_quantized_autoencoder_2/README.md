# Vector-Quantized Autoencoder
We implement a _Vector-Quantized Variational Autoencoder 2_ (_VQ-VAE-2_) for anomaly detection. 
By employing a multi-scale latent representation, the _VQ-VAE-2_ captures both global structures (top level) and local details (bottom level).

In the context of anomaly detection, this hierarchical approach allows the model to reconstruct normal patterns at different granularities. 
Anomalies often disrupt both the global consistency and local textures of an image, leading to significant reconstruction errors that are more robustly detected than in single-level architectures.

## Architecture details
The model follows a hierarchical encoder-decoder structure with Exponential Moving Average (EMA) codebook updates: 

- _Encoder_: a bottom Level captures local details, and a top level encodes the bottom-level features further to capture global features.

- _Bottleneck_: instead of standard backpropagation for the codebook, we use Exponential Moving Average (EMA) to update embedding vectors, providing more stable training and avoiding codebook collapse.
The model monitors codebook utilization to ensure a wide variety of embeddings are used to represent the normal data.

- _Decoder_: each stage incorporates residual blocks with Group Normalization to facilitate deeper feature extraction and stable gradient flow. 
A nested decoding process where the top-level latent is decoded and concatenated with the bottom-level latent before the final image reconstruction.

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
The performance of _VQ_VAE-2_ on the _Seruso_ dataset is summarized below. 

<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| VQ-VAE2         | 83.14     | 19.35   | 0.05               | 20.0              |
</div>