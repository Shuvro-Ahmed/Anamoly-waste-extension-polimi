# Detection of Anomalous Object Multi-Object Performance

The _DoAO_ is the proposed solution for high-fidelity anomaly detection and localization. 
It represents an evolution of the hierarchical _VQ-VAE-2_ architecture by integrating multi-scale consistency training and Segment Anything Model 3 refinement.

The core philosophy of _DoAO_ is to bridge the gap between global structural understanding and local textural precision while leveraging foundation models to ground anomaly heatmaps into semantically meaningful object boundaries.

## Architecture details
The model employs a hierarchical discrete latent space with specific optimizations for high-resolution industrial inspection.
The model utilizes two levels of vector quantization (Top and Bottom) with a codebook size of 1024 embeddings to prevent overfitting on normal samples and encourage more discriminative latent features.
The post-processing consists of morphological operations, cluster filtering and SAM3-refinement. 

## Installation
To install the dependencies, run: 
```bash
pip install torch torchvision transformers tqdm scipy scikit-learn pillow
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
The performance of _DoAO_ on the _Seruso_ dataset is summarized below. 

<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| DoAO            | 84.84     | 47.87     | 0.45                | 2.22               |
| DRÆM            | 43.56     | 7.90      | 0.11                | 9.09               |
</div>