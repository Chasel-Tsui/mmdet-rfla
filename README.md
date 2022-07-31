# mmdet-rfla
This is the official implementation of the ECCV2022 paper "RFLA: Gaussian Receptive based Label Assignment for Tiny Object Detection".

## Introduction
RFLA is a label assignment strategy that can replace mainstream anchor-based and anchor-free label strategies and boost their performance on tiny object detection tasks.

Abstract: Detecting tiny objects is one of the main obstacles hindering the development of object detection. The performance of generic object detectors tends to drastically deteriorate on tiny object detection tasks. In this paper, we point out that either box prior in the anchor-based detector or point prior in the anchor-free detector is sub-optimal for tiny objects. Our key observation is that the current anchor-based or anchor-free label assignment paradigms will incur many outlier tiny-sized ground truth samples, leading to detectors imposing less focus on the tiny objects. To this end, we propose a Gaussian Receptive Field based Label Assignment (RFLA) strategy for tiny object detection. Specifically, RFLA first utilizes the prior information that the feature receptive field follows Gaussian distribution. Then, instead of assigning samples with IoU or center sampling strategy, a new Receptive Field Distance (RFD) is proposed to directly measure the similarity between the Gaussian receptive field and ground truth. Considering that the IoU-threshold based and center sampling strategy are skewed to large objects, we further design a Hierarchical Label Assignment (HLA) module based on RFD to achieve balanced learning for tiny objects. Extensive experiments on four datasets demonstrate the effectiveness of the proposed methods. Especially, our approach outperforms the state-of-the-art competitors with 4.0 AP points on the AI-TOD dataset.


![demo image](figures/rfla_final.PNG)

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
Stay tuned.

## Visualization
![demo image](figures/results.gif)
