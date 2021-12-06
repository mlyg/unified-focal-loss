# Unified Focal loss
Repository for the code used in "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation".

## Source
**Update:** The paper has been accepted into Computerized Imaging and Graphics
The latest version of the preprint can be found at: https://arxiv.org/abs/2102.04525

## Description of repository contents
In this repository, please find the associated Tensorflow/Keras implementation for the following loss functions:
1. Dice loss
2. Tversky loss
3. Combo loss
4. Focal Tversky loss (symmetric and asymmetric)
5. Focal loss (symmetric and asymmetric)
7. Unified Focal loss (symmetric and asymmetric)

## Description of the Unified Focal loss
The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework. By incorporating ideas from focal and asymmetric losses, the Unified Focal loss is designed to handle class imbalance.

It can be shown that all Dice and cross entropy based loss functions described above are special cases of the Unified Focal loss:

![Overview of loss function inheritance](https://github.com/mlyg/unified-focal-loss/blob/main/Figures/Overview.png)


## Example use case 1: 2D binary datasets (CVC-ClinicDB, DRIVE, BUS2017)

The CVC-ClinicDB dataset consists of 612 frames containing polyps generated from 23 video sequences from 13 different patients using standard colonoscopy.
The DRIVE dataset consists of 40 coloured fundus photographs obtained from diabetic retinopathy screening in the Netherlands.
The BUS2017 dataset B consists of 163 ultrasound images and associated ground truth segmentations collected from the UDIAT Diagnostic Centre of the Parc Tauli Corporation, Sabadell, Spain.

The data for the CVC-ClinicDB dataset can be found at: https://polyp.grand-challenge.org/CVCClinicDB/
The data for the DRIVE dataset can be found at: https://computervisiononline.com/dataset/1105138662
The data for the BUS2017 dataset must be requested from their website: http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php

![2D binary comparison](https://github.com/mlyg/unified-focal-loss/blob/main/Figures/2D_binary.png)

Example segmentations, for each loss function for each of the three datasets. The image and ground truth are provided for reference. The false positive are highlighted in magenta, and the false negatives are highlighted in green. The yellow arrows highlight example areas where segmentation quality differs.

## Example use case 2: 3D binary (BraTS20) dataset

The BraTS20 dataset is currently the largest, publicly available and fully-annotated dataset for medical image segmentation. It comprises of 494 multimodal scans of patients with either low-grade glioma or high-grade glioblastoma. We focus on the T1 contrast-enhanced MRI scans for segmenting the enhancing tumour region.

The data for the BraTS20 dataset can be downloaded by following the instructions on their official website: https://www.med.upenn.edu/cbica/brats2020/data.html

![BraTS20 comparison](https://github.com/mlyg/unified-focal-loss/blob/main/Figures/3D_binary.png)

Axial slice from an example segmentation for each loss function for the BraTS20 dataset. The image and ground truth are provided for reference. The false positive are highlighted in magenta, and the false negatives are highlighted in green. The yellow arrows highlight example areas where segmentation quality differs.

## Example use case 3: 3D multiclass (KiTS19) dataset

The KiTS19 dataset consists of 300 arterial phase abdominal CT scans. These are from patients who underwent partial removal of the tumour and surrounding kidney or complete removal of the kidney including the tumour at the University of Minnesota Medical Center, USA.

The data for the KiTS19 dataset can be downloaded from their official github repository: https://github.com/neheller/kits19

![KiTS19 comparison](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/3D_multiclass.png)

Axial slice from an example segmentation for each loss function for the KiTS19 dataset. The image and ground truth are provided for reference. The red contour corresponds to the kidneys, and the blue contour to the tumour.

## How to use the Unified Focal loss

For our experiments, we make use of the Medical Image Segmentation with Convolutional Neural Networks (MIScnn) open-source python library: 
https://github.com/frankkramer-lab/MIScnn

To use each loss function, they can be called in model.compile where the hyperparameters can be specified (otherwise default values used):

For example:
model.compile(loss = asym_unified_focal_loss(), ...)
