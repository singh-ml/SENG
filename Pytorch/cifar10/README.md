# Pytorch-CIFAR10-SENG Example

## Introduction

The Sketchy Empirical Natural Gradient Methods (**SENG**) is an algorithm for solving large-scale deep learning problems. It utilizes efficient sketching methods to reduce the computational and memory complexity of the Empirical Natural Gradient method.

Currently, **SENG** supports the convolution layer, batch normalization layer and fully-connected layer. 

This repository provides an **Pytorch** implementation of **SENG** in the manuscript:
- Sketchy Empirical Natural Gradient Methods for Deep Learning (https://arxiv.org/pdf/2006.05924.pdf).

## Requirements
* GPU with at least 8GB of memory.
* Python (>= 3.6)
* [pytorch (>= 1.5.0)](http://pytorch.org/)
* torchvision


## Usage

- First install all the requirements and download the source code. 

- An example for training **ResNet-18** on **CIFAR10** with one GPUs:
  ```python
  python main_seng.py --epoch 65 --arch 'resnet18' --lr-decay-epoch 70 --damping 1.0 --trainset 'cifar10' --lr 0.05 --weight-decay 5e-4 --lr-scheme 'cosine' --gpu 0| tee your/store/file 
  ```
  

- An example for training **VGG18_bn** on **CIFAR10** with one GPUs:

  ```python
  python main_seng.py --epoch 65 --arch 'vgg16_bn' --lr-decay-epoch 70 --damping 2.0 --trainset 'cifar10' --lr 0.05 --weight-decay 5e-4 --lr-scheme 'cosine' --gpu 0| tee your/store/file 
  ```

## The Authors

If you have any bug reports or comments, please feel free to email one of the authors:

* Minghan Yang, yangminghan at pku.edu.cn
* Dong Xu, taroxd at pku.edu.cn
* Zaiwen Wen, wenzw at pku.edu.cn

## License

This package is released under [GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html).