# Discriminative Reconstruction-based Anomaly Embedding 
We adopt _DRÆM_, a reconstructive-discriminative network designed specifically for unsupervised anomaly detection. 
_DRÆM_ operates by learning to reconstruct images that have been artificially augmented with simulated anomalies. 
It consists of two sub-networks: a reconstructive sub-network that attempts to recover the original image, and a discriminative sub-network that generates a pixel-wise anomaly map by comparing the original and reconstructed images.
This approach allows the model to learn a robust representation of normality while simultaneously being trained to localize deviations from it.

## Architecture details
The model is implemented using the Anomalib library, utilizing a architecture that focuses on embedding reconstruction errors.
The model is composed by two sub-networks: 

- _Reconstruction_: an encoder-decoder structure (similar to a U-Net) that learns to map anomalous-looking inputs back to their normal counterparts.

- _Localization_: a discriminative network that takes the concatenation of the input and the reconstructed image to output a high-resolution anomaly heat map.

The network operates on RGB images resized to 256×256.
For the post-processing we apply a Gaussian filter to the error map to reduce pixel noise, and then we generate the anomaly map using a 90th percentile threshold.

## Installation
This implementation requires the anomalib library and its dependencies.

```bash
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install .
```
> _Citation_: Zavrtanik, V., Kristan, M., & Skočaj, D. (2021). DRAEM - A discriminative reconstruction-based learned representation for surface anomaly detection. ICCV. https://github.com/openvinotoolkit/anomalib

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
The performance of _DRÆM_ on the _Seruso_ dataset is summarized below. 

<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| DRÆM            | 43.56     | 7.90      | 0.11                | 9.09                |
</div>