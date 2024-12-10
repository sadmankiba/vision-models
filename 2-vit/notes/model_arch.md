# Model architectures

Input 128 x 128 x 3

## SimpleNet

Conv : 64 7x7 s2 p3 ReLU . Output: 64 64x64
Max Pooling: 3x3 s2 p1. Output: 64 32x32

Conv : 64 1x1 s1 p0 ReLU. Output: 64 32x32
Conv : 64 3x3 s1 p1 ReLU. Output: 64 32x32
Conv : 256 1x1 s1 p0 ReLU. Output: 256 32x32
Max Pooling: 3x3 s2 p1. Output: 256 16x16

Conv : 128 1x1 s1 p0 ReLU. Output: 128 16x16
Conv : 128 3x3 s1 p1 ReLU. Output: 128 16x16
Conv : 512 1x1 s1 p0 ReLU. Output: 512 16x16

AvgPool: 1x1. Output: 512
FC: 512x100. Output: 100

## ResNet18

Conv : 64 7x7 s2 p3 BN ReLU. Output: 64 64x64
Max Pooling: 3x3 s2 p1. Output: 64 32x32

Layer1
Conv : 64 3x3 s1 p1 BN ReLU. Output: 64 32x32
Conv : 64 3x3 s1 p1 BN. Output: 64 32x32
Conv : 64 3x3 s1 p1 BN ReLU. Output: 64 32x32
Conv : 64 3x3 s1 p1 BN. Output: 64 32x32

Layer2
Conv : 128 3x3 s2 p1 BN ReLU. Output: 128 16x16
Conv : 128 3x3 s1 p1 BN. Output: 128 16x16
Conv : 128 3x3 s1 p1 BN ReLU. Output: 128 16x16
Conv : 128 3x3 s1 p1 BN. Output: 128 16x16

Layer3
Conv : 256 3x3 s2 p1 BN ReLU. Output: 256 8x8
Conv : 256 3x3 s1 p1 BN. Output: 256 8x8
Conv : 256 3x3 s1 p1 BN ReLU. Output: 256 8x8
Conv : 256 3x3 s1 p1 BN. Output: 256 8x8

Layer4
Conv : 512 3x3 s2 p1 BN ReLU. Output: 512 4x4
Conv : 512 3x3 s1 p1 BN. Output: 512 4x4
Conv : 512 3x3 s1 p1 BN ReLU. Output: 512 4x4
Conv : 512 3x3 s1 p1 BN. Output: 512 4x4

AvgPool: 1x1. Output: 512
Linear: 512 x 1000

## CustomNet

Conv : 64 7x7 s2 p3 BN ReLU . Output: 64 64x64
Max Pooling: 3x3 s2 p1. Output: 64 32x32

Conv : 64 1x1 s1 p0 BN ReLU. Output: 64 32x32
Conv : 64 3x3 s1 p1 BN ReLU. Output: 64 32x32
Conv : 256 1x1 s1 p0 BN ReLU. Output: 256 32x32
Max Pooling: 3x3 s2 p1. Output: 256 16x16

Conv : 128 1x1 s1 p0 BN ReLU. Output: 128 16x16
Conv : 128 3x3 s1 p1 BN ReLU. Output: 128 16x16
Conv : 512 1x1 s1 p0 BN ReLU. Output: 512 16x16

AvgPool: 1x1. Output: 512
FC: 512x100. Output: 100

## CustomNet2

Conv : 64 7x7 s2 p3 BN ReLU . Output: 64 64x64
Max Pooling: 3x3 s2 p1. Output: 64 32x32

Conv : 64 1x1 s1 p0 BN ReLU. Output: 64 32x32
Conv : 64 3x3 s1 p1 BN ReLU. Output: 64 32x32
Conv : 64 3x3 s1 p1 BN. Output: 64 32x32

Conv : 128 1x1 s1 p0 BN ReLU. Output: 128 32x32
Conv : 128 3x3 s2 p1 BN ReLU. Output: 128 16x16
Conv : 128 3x3 s1 p1 BN. Output: 128 16x16

Conv : 256 1x1 s1 p0 BN ReLU. Output: 128 16x16
Conv : 256 3x3 s2 p1 BN ReLU. Output: 256 8x8
Conv : 256 3x3 s1 p1 BN. Output: 256 8x8

Conv : 512 1x1 s1 p0 BN ReLU. Output: 512 8x8
Conv : 512 3x3 s2 p1 BN ReLU. Output: 512 4x4
Conv : 512 3x3 s1 p1 BN. Output: 512 4x4

AvgPool: 1x1. Output: 512
FC: 512x100. Output: 100

Difference from CustomNet
* Kept only 1 max pooling at the very beginning. 
* Followed ResNet architecture to have 4 blocks each with 3 convolution layers. In each block, the first two layers have ReLU, but last layer does not. In all blocks other than first block, the middle layer has stride 2 to downsample. In all blocks, the first layer has 1x1 kernels (similar to SimpleNet and CustomNet) and other layers have 3x3 kernels.   
* Increased number of kernels gradually (64, 128, 256, 512). Made network deeper.
* Final average pooling on 4x4. 

## Conv Results

After 60 epochs. SimpleNet reached 44% accuracy. ResNet18 finetuning reached 52.3% accuracy. CustomNet reached 46% accuracy. CustomNet2 reached 48.6% accuracy.


## ViT

Arch1: 4 blocks, 4 heads, dim 192. bs 256. lr 0.01, wd 0.05. 
Arch2: 6 blocks, 4 heads, dim 768. bs 256. lr 0.001, wd 0.005. 

Arch 1 achieves 28% top-1 after 90 epochs. Arch2 fails to learn. After 75 epochs, top-1 acc is 7%.  


## Miscellaneous

Details about an image in command line
```sh
sudo apt update
sudo apt install imagemagick
identify path/to/your/image.jpg
```