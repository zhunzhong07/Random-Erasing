# Random Erasing Data Augmentation
===============================================================
### This code has the source code for the paper "[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)".

If you find this code useful in your research, please consider citing:

    @article{zhong2017random,
    title={Random Erasing Data Augmentation},
    author={Zhong, Zhun and Zheng, Liang and Kang, Guoliang and Li, Shaozi and Yang, Yi},
    journal={arXiv preprint arXiv:1708.04896},
    year={2017}
    }

Thanks for [Marcus D. Bloice](https://github.com/mdbloice), Marcus D. Bloice reproduces our method in [Augmentor](http://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.random_erasing). [Augmentor](http://augmentor.readthedocs.io/en/master/index.html) is an image augmentation library in Python for machine learning.

| Original image                                                                                             | Random Erasing                                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| ![Original](https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/UsageGuide/city-road-street-italy-scaled.jpg) | ![Original](https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/UsageGuide/city-road-street-italy-animation.gif) |

### Other re-implementations

[\[Python Augmentor\]](http://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.random_erasing)

[\[CamStyle\]](https://github.com/zhunzhong07/CamStyle)

[\[Keras\]](https://github.com/yu4u/cutout-random-erasing)

[\[Person_reID_baseline + Random Erasing + Re-ranking\]](https://github.com/layumi/Person_reID_baseline_pytorch)

### Installation

Requirements for Pytorch （see [Pytorch](http://pytorch.org/) installation instructions）

### Examples:

#### CIFAR10

ResNet-20 baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --p 0.5
    ```

#### CIFAR100

ResNet-20 baseline on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20 --p 0.5
    ```

#### Fashion-MNIST


ResNet-20 baseline on Fashion-MNIST：
    ```
    python fashionmnist.py --dataset fashionmnist --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on Fashion-MNIST：
    ```
    python fashionmnist.py --dataset fashionmnist --arch resnet --depth 20 --p 0.5
    ```

### Other architectures

For ResNet： 
    ```
    --arch resnet --depth (20， 32， 44， 56， 110)
    ```

For WRN：
    ```
    --arch wrn --depth 28 --widen-factor 10
    ```

### Our results

You can reproduce the results in our paper:

| |  CIFAR10 | CIFAR10| CIFAR100 | CIFAR100| Fashion-MNIST | Fashion-MNIST|
| -----   | -----  | ----  | -----  | ----  | -----  | ----  |
|Models |  Base. | +RE | Base. | +RE | Base. | +RE |
|ResNet-20 |  7.21 | 6.73 | 30.84 | 29.97 | 4.39 | 4.02 |
|ResNet-32 |  6.41 | 5.66 | 28.50 | 27.18 | 4.16 | 3.80 |
|ResNet-44 |  5.53 | 5.13 | 25.27 | 24.29 | 4.41 | 4.01 |
|ResNet-56 |  5.31 | 4.89| 24.82 | 23.69 | 4.39 | 4.13 |
|ResNet-110 |  5.10 | 4.61 | 23.73 | 22.10 | 4.40 | 4.01 |
|WRN-28-10 |  3.80 | 3.08 | 18.49 | 17.73 | 4.01 | 3.65 |

### NOTE THAT, if you use the latest released Fashion-MNIST, the performance will slightly lower than the results reported in our paper. Please refer to the [issue](https://github.com/zhunzhong07/Random-Erasing/issues/9).



If you have any questions about this code, please do not hesitate to contact us.

[Zhun Zhong](http://zhunzhong.site)

[Liang Zheng](http://liangzheng.com.cn)
