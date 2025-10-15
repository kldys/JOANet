# JOANet
JOANet: An Integrated Joint Optimization Architecture Making Medical Image Segmentation Really Helped by Super-resolution Pre-processing
## Overview
Conventional computer vision pipelines typically treat low-level enhancement and high-level semantic tasks as isolated processes, focusing on optimizing enhancement for perceptual quality rather than computational utility, neglecting semantic task requirements. To bridge this gap, this paper proposes an integrated joint optimization architecture that aligns the objectives of enhancement tasks with the practical needs of semantic tasks. Specifically, the architecture ensures that medical image segmentation (the semantic task) benefits directly from super-resolution pre-processing (the enhancement task). This integrated architecture fundamentally differs from conventional sequential frameworks by enabling joint training of super-resolution and segmentation networks. Guided by its own content reconstruction loss and semantic loss transferred from segmentation, the super-resolution network prioritizes semantically significant regions for segmentation-driven reconstruction.

<div align="center">
<img src="fig/1.png" width="600px">
</div>

*Figure 1: A higher PSNR value in an input image does not guarantee improved segmentation results. (a) Original high-resolution image and its standard segmentation annotation. (b) Segmentation result of the low-resolution image. (c) Segmentation result of the super-resolution (SR) image reconstructed by a network trained independently. (d) Segmentation result of the super-resolution image reconstructed by a network trained using the joint training framework.*

<div align="center">
<img src="fig/2.png" width="600px">
</div>

*Figure 2: Composition of loss functions in JOANet.*

## Prerequisites

- Python 3.10+
- PyTorch 2.0.1+
- NVIDIA GPU with ≥ 40GB memory (A100 recommended)

## Training Guide
The following datasets were used in this article:

**1. Synapse Multi-organ Segmentation Dataset**: 30 patients' abdominal CT images from MICCAI 2015 Multi-Organ CT Annotation Challenge.

**2. Automated Cardiac Diagnosis Challenge Dataset**: Cardiac MRI scans with left ventricle, right ventricle, and myocardium annotations.

**3. Promise12 Dataset**: Prostate MRI images from multiple centers with diverse acquisition protocols.

**4. Brain Tumour Segmentation Dataset**: Real clinical T1-weighted MRI with brain tumor annotations.

### Data Generation Process
During training, we generate low-resolution training images by applying **bicubic downsampling** to the original high-resolution images. This process simulates real-world low-quality medical imaging conditions.

### Complete Training Dataset Requirements
A complete training dataset must contain three key components:
1. **Low-resolution images** - Generated via bicubic downsampling
2. **Original high-resolution images** - Used as reconstruction targets
3. **Segmentation masks** - Ground truth labels for semantic tasks

###  Configure Training Paths
Modify the path settings in `train.py` to match your dataset locations and strart training with 
```Bash
python train.py
```
## Testing Guide
A pretrained model on the BTS dataset is available in the `model/` folder for testing:

```bash
python test.py
```
