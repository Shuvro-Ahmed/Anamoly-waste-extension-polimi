# Autoencoder

We adopt an _Autoencoder_ based on a _U-Net_ architecture to perform image reconstruction for anomaly detection.
The model learns to reconstruct normal image regions, and anomalies are identified through reconstruction errors.
Compared to a plain _Autoencoder_, the _U-Net_ structure preserves spatial information by combining multi-scale features through skip connections, leading to more accurate localization of anomalous areas.

## Architecture details
The architecture follows a symmetric encoder–decoder U-Net design with skip connections between corresponding resolution levels:

- _Encoder_: the encoder progressively reduces the spatial resolution while increasing the number of feature channels.
Each stage consists of a ConvBlock with two 3×3 convolutions, Batch Normalization, and ReLU activation.
Downsampling is performed using 2×2 max-pooling.
Feature depth doubles at each level.

- _Decoder_: the decoder reconstructs the image by progressively increasing the spatial resolution.
Upsampling is performed using transposed convolutions.
At each level, the upsampled features are concatenated with the corresponding encoder features (skip connections).
Each concatenation is followed by a ConvBlock.
A final 1×1 convolution maps the 32-channel feature map to RGB. 
A Sigmoid activation is applied to constrain the reconstructed pixel values to the range [0,1].

## Usage 
To train the model, configure the desired hyper-parameters in the training script and run:
```bash
python3 training.py
```

Move the selected checkpoint into the `checkpoint` folder and run:
```bash
python3 inference.py
```
The script will generate a directory containing the reconstructed images and anomaly maps.

## Results
The performance of _Autoencoder_ on the _Seruso_ dataset is summarized below. 

<div align="center">

| Method          | AUROC (%) | IoU (%) | Inference time (s) | Images per second |
|-----------------|-----------|---------|--------------------|-------------------|
| Autoencoder     | 81.04     | 17.46   | 0.06               | 16.7              |

</div>