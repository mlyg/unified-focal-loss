# Mixed Focal loss
Repository for the code used in "A Mixed Focal Loss Function for Handling Class Imbalanced Medical Image Segmentation".
The paper can be found at: https://arxiv.org/abs/2102.04525

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
The Mixed Focal loss is defined as the linear weighted sum of the modified Focal loss and modified Focal Dice loss. 

## How to use the Mixed Focal loss
For our experiments, we make use of the Medical Image Segmentation with Convolutional Neural Networks (MIScnn) open-source python library: 
https://github.com/frankkramer-lab/MIScnn

The Mixed Focal loss can be passed directly as a loss function into model.compile:

model.compile(loss =  mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75), ...)

The parameters are defined as follows:
1. weight: lambda variable controlling weight given to Focal Tversky loss and Focal loss. If weighting is desired, this should be set between 0 and 1 with weight > 0.5 assigning more weight to Focal Dice loss than Focal loss. The default is to assign equal weights to both Focal Tversky loss and Focal loss. 
2. alpha: a vector of weights associated with each class. **The length of the vector must match the number of classes**. For example, for 2 class alpha = [x,y], for 3 classes alpha = [x,y,z]. The default is for equal weighting ('None'). 
3. beta: a variable controlling the relative contirbution of false positive and false negative predictions on the modified Focal loss. Beta > 0.5 penalises false negatives more than false positives. The default is for equal focus ('None').
4. delta: a variable controlling the relative contribution of false positive and false negative predictions on the modified Focal Dice loss. Delta > 0.5 penalises false negatives more than false positives. The default is the Tversky Index (delta = 0.7). 
4. gamma_f: Focal loss focal parameter controls degree of down-weighting of easy examples. The default is gamma_f = 2. 
5. gamma_ft: Focal Tversky loss focal parameter controls degree of down-weighting of easy examples. The default is gamma_ft = 0.75. 

## How to use the other loss functions in this repository
1. Dice loss and Tversky loss do not have any modifiable parameters and should be passed directly to model.compile without invocation:
i.e. model.compile(loss=dice_loss) or model.compile(loss=tversky_loss)

2. Cosine Tversky loss has a single focal parameter gamma that controls the degree of down-weighting of easy examples. Default is gamma = 1 i.e. no weighting. 

3. Combo loss has two parameters, alpha and beta:
a) alpha: variable that controls weighting of dice and cross-entropy loss
b) beta: a variable controlling the relative contirbution of false positive and false negative predictions on the modified Focal loss. Beta > 0.5 penalises false negatives more than false positives. The default is for equal focus ('None').
The default is for alpha = beta = 0.5
