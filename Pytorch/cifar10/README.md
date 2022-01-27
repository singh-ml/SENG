# Example of SENG in Pytorch for CIFAR10

## Introduction

This folder provides a **Pytorch** implementation of **SENG** on CIFAR10 dataset:

## Requirements
* GPU with at least 8GB of memory.
* Python (>= 3.6)
* [pytorch (>= 1.5.0)](http://pytorch.org/)
* torchvision


## Usage

- First install all the requirements and download the source code. Put the the dataset on any place, e.g., 'your/data/path'.

- An example for training **ResNet18** on **CIFAR10** with one GPU:
  ```python
  python main_seng.py --epoch 65 --arch 'resnet18' --lr-decay-epoch 70 --damping 1.0 --trainset 'cifar10' --datadir your/data/path --lr 0.05 --weight-decay 5e-4 --lr-scheme 'cosine' --gpu 0| tee your/store/file 
  ```
  

- An example for training **VGG16_bn** on **CIFAR10** with one GPU:

  ```
  python main_seng.py --epoch 100 --arch 'vgg16_bn'  --damping 2.0 --trainset 'cifar10' --datadir your/data/path --lr 0.05 --weight-decay 5e-4 --gpu 0| tee  your/store/file 
  ```

## The Authors

If you have any bug reports or comments, please feel free to email one of the authors:

* Minghan Yang, yangminghan at pku.edu.cn
* Dong Xu, taroxd at pku.edu.cn
* Zaiwen Wen, wenzw at pku.edu.cn

## License

This package is released under [GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html).
