# Generalised Focal loss
Repository for the code used in "Generalised Focal Loss: Unifying Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation".

## Source
The preprint version of the paper can be found at: https://arxiv.org/abs/2102.04525

## Description of repository contents
In this repository, please find the associated Tensorflow/Keras implementation for the following loss functions:
1. Dice loss
2. Tversky loss
3. Combo loss
4. Focal Tversky loss
5. Focal loss
7. Hybrid Focal loss (previously called Mixed Focal loss)
8. Generalised Focal loss

## Description of the Generalised Focal loss
The Generalised Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework. By incorporating ideas from focal and asymmetric losses, the Generalised Focal loss is designed to handle class imbalance, and we achieve state-of-the-art performance using this loss function. 

It can be shown that all Dice and cross entropy based loss functions described above are special cases of the Generalised Focal loss:

![Overview of loss function inheritance](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/Overview.png)

## Example use case 1: Breast UltraSound 2017 (BUS2017) dataset

The BUS2017 dataset B consists of 163 ultrasound images and associated ground truth segmentations collected from the UDIAT Diagnostic Centre of the Parc Tauli Corporation, Sabadell, Spain.

The data for the BUS2017 dataset must be requested from their website: http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php

In our paper, we use the standard U-Net with the Generalised Focal loss and compare with state-of-the-art solutions:

![BUS2017 comparison](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/BUS2017_table.png)


## Example use case 2: Brain Tumour Segmentation 2020 (BraTS20) dataset

The BraTS20 dataset is currently the largest, publicly available and fully-annotated dataset for medical image segmentation. It comprises of 494 multimodal scans of patients with either low-grade glioma or high-grade glioblastoma. We focus on the T1 contrast-enhanced MRI scans for segmenting the enhancing tumour region.

The data for the BraTS20 dataset can be downloaded by following the instructions on their official website: https://www.med.upenn.edu/cbica/brats2020/data.html

In our paper, we use the standard U-Net with the Generalised Focal loss and compare with state-of-the-art solutions:

![BraTS20 comparison](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/BraTS20_table.png)

## Example use case 3: Kidney Tumour Segmentation 2019 (KiTS19) dataset

The KiTS19 dataset consists of 300 arterial phase abdominal CT scans. These are from patients who underwent partial removal of the tumour and surrounding kidney or complete removal of the kidney including the tumour at the University of Minnesota Medical Center, USA.

The data for the KiTS19 dataset can be downloaded from their official github repository: https://github.com/neheller/kits19

In our paper, we use the standard U-Net with the Generalised Focal loss and compare with state-of-the-art solutions:

![KiTS19 comparison](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/KiTS19_table.png)


## Example segmentations using different loss functions

![Example comparison](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/Example_outputs.png)

Image segmentations generated from the BUS2017 (top), BraTS20 (middle) and KiTS19 (bottom) datasets using (a) ground truth, (b) Focal loss, (c) Dice loss, (d) Tversky loss, (e) Focal Tversky loss, (f) Combo loss, (g) Hybrid Focal loss and (h) Generalised Focal loss. For BUS2017 and BraTS20 datasets, the breast lesion and enhancing tumour region are highlighted in red respectively. For the KiTS19 dataset, the kidney is highlighted in red and the tumour in blue. A magnified contour of the segmentation is provided in the top right-hand corner of each image.


## How to use the Generalised Focal loss

For our experiments, we make use of the Medical Image Segmentation with Convolutional Neural Networks (MIScnn) open-source python library: 
https://github.com/frankkramer-lab/MIScnn

The Generalised Focal loss can be passed directly as a loss function into model.compile:

model.compile(loss = generalised_focal_loss(weight=0.5, delta=0.7, gamma=0.2), ...)
