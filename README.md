# CIFAR100

The purpose of the project was to create a neural network to classify images from CIFAR100 dataset into 100 classes. Three neural networks were created to see different approaches and decide which neural networks performed best. 

## CIFAR100.py

This file contains a neural network made with PyTorch library. Neural network has 7 layers - first four are convolutional layers (each of them has 3x3 kernel size and is activated by ReLU function). Then the data is flattened. Next two layers are linear layers with ReLU activation, and the last linear layer is activated by LogSoftmax.

## CIFAR100lightning.py

The second approach was pretty much the same. The only change was to convert neural network to PyTorch Lightning. 

## CIFAR100resnet.py

The last approach was to made the neural network in completely different way. This time it is a resnet having a residual blocks containing two convolutional layers (each of them has 3x3 kernel size) and each of them is activated by gelu function. The first layer is conventional convolutional layer with relu activation function and then the residual block is repeated 3 times. Next there are two linear layers activated by glu function after each of them there is a dropout layer. The last layer is still linear and activated by LogSoftmax.

## Summary

The best performing model was the resnet, which is expected as resnets have been seen to outperform classic convnets most of the time. 

## Technologies
* Python - version 3.8.3
* Numpy - version 1.18.5
* Pandas - version 1.0.5
* PyTorch - version 1.7.0 CUDA 10.1
* PyTorch-Lightning - version 1.0.6
