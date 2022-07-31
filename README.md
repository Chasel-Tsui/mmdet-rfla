# mmdet-rfla
This is the official implementation of the ECCV2022 paper "RFLA: Gaussian Receptive based Label Assignment for Tiny Object Detection".

## Introduction
RFLA is a label assignment strategy that can replace mainstream anchor-based and anchor-free label strategies and boost their performance on tiny object detection tasks.

Abstract:

![demo image](figures/structure.png)

## Installation and Get Started

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Install TODbox:

Note that this repository is based on the [MMDetection](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/Chasel-Tsui/mmdet-rfla.git
pip install -r requirements/build.txt
python setup.py develop

## Main Results

## Visualization
![demo image](figures/results.gif)
