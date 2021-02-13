# Mixed Focal loss
Repository for the code used in "A Mixed Focal Loss Function for Handling Class Imbalanced Medical Image Segmentation".

## Source
The preprint version of the paper can be found at: https://arxiv.org/abs/2102.04525

## Description of repository contents
In this repository, please find the associated Keras implementation for the following loss functions:
1. Dice loss
2. Focal loss
3. Tversky loss
4. Focal Tversky loss
5. Cosine Tversky loss
6. Combo loss
7. Mixed Focal loss

## Description of the Mixed Focal loss
The Mixed Focal loss is a new compound loss function and defined as the linear weighted sum of the modified Focal loss and modified Focal Dice loss. The tunable parameters enable optimisation for recall-precision balance. and to deal with class imbalanced image segmentation with its focal parameter.

The Mixed Focal loss inherits properties from variants of both the Dice loss and cross-entropy loss:

![Overview of loss function inheritance](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/Overview_loss_functions.png)

## Example use case 1: Kidney Tumour Segmentation 2019 (KiTS19) dataset

The KiTS19 dataset consists of 300 arterial phase abdominal CT scans. These are from patients who underwent partial removal of the tumour and surrounding kidney or complete removal of the kidney including the tumour at the University of Minnesota Medical Center, USA.

The data for the KiTS19 dataset can be downloaded from their official github repository: https://github.com/neheller/kits19

In our paper, we compare 7 loss functions using the KiTS19 dataset and generate the following segmentations as a result:

![Segmentations generated using KiTS19 dataset](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/KiTS19_segmentation.png)

(a) ground truth, (b) Focal loss, (c) Dice loss, (d) Tversky loss, (e) Cosine Tversky loss, (f) Focal Tversky loss, (g) Combo loss, (h) Mixed Focal loss. The kidney is highlighted in red and the tumour in blue. A magnified contour of the segmentation is provided in the top right-hand corner of each image.


## Example use case 2: Brain Tumour Segmentation 2020 (BraTS20) dataset

The BraTS20 dataset is currently the largest, publicly available and fully-annotated dataset for medical image segmentation. It comprises of 494 multimodal scans of patients with either low-grade glioma or high-grade glioblastoma. We focus on the T1 contrast-enhanced MRI scans for segmenting the enhancing tumour region.

The data for the BraTS20 dataset can be downloaded by following the instructions on their official website: https://www.med.upenn.edu/cbica/brats2020/data.html

In our paper, we compare the top three performing loss functions using the BraTS20 dataset and generate the following segmentations as a result:

![Segmentations generated using BraTS20 dataset](https://github.com/mlyg/mixed-focal-loss/blob/main/Figures/BraTS20_segmentation.png)

(a) ground truth, (b) Focal Tversky loss, (c) Combo loss and (d) Mixed Focal loss. Tumour is highlighted in red. A magnified contour of the segmentation is provided in (e-h) below each respective image.

## How to use the Mixed Focal loss
For our experiments, we make use of the Medical Image Segmentation with Convolutional Neural Networks (MIScnn) open-source python library: 
https://github.com/frankkramer-lab/MIScnn

The Mixed Focal loss can be passed directly as a loss function into model.compile:

model.compile(loss =  mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75), ...)

The parameters are defined as follows:
1. weight: lambda variable controlling weight given to the modified Focal loss and Focal Dice loss. If weighting is desired, this should be set between 0 and 1 with weight > 0.5 assigning more weight to Focal Dice loss than Focal loss. The default is to assign equal weights to both Focal Tversky loss and Focal loss. 
2. alpha: a vector of weights associated with each class. **The length of the vector must match the number of classes**. For example, for 2 class alpha = [x,y], for 3 classes alpha = [x,y,z]. The default is for equal weighting ('None'). 
3. beta: a variable controlling the relative contirbution of false positive and false negative predictions on the modified Focal loss. Beta > 0.5 penalises false negatives more than false positives. The default is for equal focus ('None').
4. delta: a variable controlling the relative contribution of false positive and false negative predictions on the modified Focal Dice loss. Delta > 0.5 penalises false negatives more than false positives. The default is the Tversky Index (delta = 0.7). 
4. gamma_f: Focal loss focal parameter controls degree of down-weighting of easy examples. The default is gamma_f = 2. 
5. gamma_fd: Focal Dice loss focal parameter controls degree of down-weighting of easy examples. The default is gamma_fd = 0.75. 

## How to use the other loss functions in this repository
1. Dice loss and Tversky loss do not have any modifiable parameters and should be passed directly to model.compile without invocation:
i.e. model.compile(loss=dice_loss) or model.compile(loss=tversky_loss)

2. Cosine Tversky loss has a single focal parameter gamma that controls the degree of down-weighting of easy examples. Default is gamma = 1 i.e. no weighting. 

3. Combo loss has two parameters, alpha and beta:
a) alpha: variable that controls weighting of Dice and cross-entropy loss
b) beta: a variable controlling the relative contirbution of false positive and false negative predictions on the modified Focal loss. Beta > 0.5 penalises false negatives more than false positives. The default is for alpha = beta = 0.5
