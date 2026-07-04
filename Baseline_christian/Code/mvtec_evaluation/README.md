# Detection of Anomalous Object Single-Object Performance

This is the proposed _DoAO_ framework evaluated on the _MVTec AD_ dataset to test the single-object anomaly detection performance. 
The _DoAO_ approach introduces multi-scale consistency by training on both full-resolution images and corner-based patches. 
This dual training regime ensures the model captures global structural integrity and fine-grained local textures simultaneously.

## Architecture details
The model employs a hierarchical discrete latent space with specific optimizations for high-resolution industrial inspection.
The model utilizes two levels of vector quantization (Top and Bottom) with a reduced codebook size of 256 embeddings to prevent overfitting on normal samples and encourage more discriminative latent features.
The post-processing consists of morphological operations and cluster filtering. 

## Installation
To install the dependencies, run: 
```bash
pip install torch torchvision tqdm scipy scikit-learn pillow numpy
```

# Usage 
The training script iterates through all object categories in the _MVTec AD_ dataset. 
It optimizes a combined loss of reconstruction, quantization loss, and multi-scale consistency.
To train the model on _MVTec AD_ dataset, run:

```bash
python3 training.py
```

To evaluate the model using a saved checkpoint, run:

```bash
python3 inference.py
```

# Results
The performance of _DoAO_ on the _MVTec AD_ dataset is summarized below. 

<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| DoAO            | 73.24     | 18.42     | 0.1                | 10                |
| DRÆM            | 97.9     | 60.3      | -                | -               |
</div>