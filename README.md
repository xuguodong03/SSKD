# SSKD
This repo is the implementation of paper Knowledge Distillation Meets Self-Supervision.

## Prerequisite
This repo is tested with Ubuntu 16.04.5, Python 3.7, PyTorch 1.5.0, CUDA 10.2.
Make sure to install pytorch, torchvision, tensorboardX, numpy before using this repo.

## Running

### Teacher Training
An example of teacher training is:
```
python teacher.py --arch wrn_40_2 --lr 0.05 --gpu-id 0
```
where you can specify the architecture via flag `--arch`

### Student Training
An example of student training is:
```
python student.py --t-path ./experiments/teacher_wrn_40_2_seed0/ --s-arch wrn_16_2 --lr 0.05 --gpu-id 0
```
The meanings of flags are:
> `--t-path`: teacher's checkpoint path. Automatically search the checkpoint containing 'best' keyword in its name.

> `--s-arch`: student's architecture.

All the commands can be found in `command.sh`

## Results

### Similar-Architecture

| Teacher <br> Student | wrn-40-2 <br> wrn-16-2 | wrn-40-2 <br> wrn-40-1 | resnet56 <br> resnet20 | resnet32x4 <br> resnet8x4 |  vgg13 <br> vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    75.61 <br> 73.26    |    75.61 <br> 71.98    |    72.34 <br> 69.06    |     79.42 <br> 72.50     | 74.64 <br> 70.36 |
| KD | - | - | - | - | - |
| FitNet | - | - | - | - | - |
| AT | - | - | - | - | - |
| SP | - | - | - | - | - |
| CC | - | - | - | - | - |
| VID | - | - | - | - | - |
| RKD | - | - | - | - | - |
| PKT | - | - | - | - | - |
| AB | - | - | - | - | - |
| FT | - | - | - | - | - |
| FSP | - | - | - | - | - |
| NST | - | - | - | - | - |
| CRD | - | - | - | - | - |
| SSKD | - | - | - | - | - |

### Cross-Architecture
