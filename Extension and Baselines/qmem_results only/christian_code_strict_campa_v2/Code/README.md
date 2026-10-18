# Detection Of Anomalous Objects

<div align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python"> 
    <img src="https://img.shields.io/badge/PyTorch-5C3EE8?logo=pytorch&logoColor=white&style=for-the-badge" alt="PyTorch">
    <img src="https://img.shields.io/badge/Transformers-FF6F00?logo=huggingface&logoColor=white&style=for-the-badge" alt="Transformers">
</div>

This project focuses on developing a modern model capable of autonomously sorting waste in top-down images from a recycling center. 
The model is trained only on images of normal objects, aiming to detect anomalous or defective items.

We start from classical unsupervised anomaly detection, which typically assumes a single object with some defect and demonstrates its limitations in multi-object scenarios. 
We then propose a novel approach designed to detect entire anomalous objects within a single image, which we also validate on traditional anomaly detection datasets.


The repository is organized as follows: 
- [Baselines](https://github.com/Chri060/thesis-2025-rossi/tree/main/baselines): contains the baseline models used in this research, including autoencoders, diffusion models, and the newest unsupervised anomaly detection models.
- [Proposed solution](https://github.com/Chri060/thesis-2025-rossi/tree/main/proposed_solution): contains our novel approach for multi-object unsupervised anomaly detection.
- [MvTec evaluation](https://github.com/Chri060/thesis-2025-rossi/tree/main/mvtec_evaluation): results of our proposed model evaluated on the MVTec dataset.
- [Deliverables](https://github.com/Chri060/thesis-2025-rossi/tree/main/deliverables): includes the research thesis and the final presentation.

The datasets used are Seruso and MvTec detailed below. 
<div align="center">

| Dataset                                                                             | Description                                                 |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Standard benchmark for single-object anomaly detection      |
| [Seruso]([missing.it](https://ug.link/dh2300-aiXU/filemgr/share-download/?id=9bae0f10de524231b5e570331dca8141)) | Custom multi-object dataset collected in a recycling center |

</div>

## Installation
The main folder contains a `requirements.txt` file listing the necessary dependencies. 
To install them:

```bash
pip install -r requirements.txt
```

> **_NOTE:_** Some dependencies may need manual installation or downloads from other repositories. 
> Instructions are provided in each folder's README.

## Authors

- [Christian Rossi](https://github.com/Chri060)

