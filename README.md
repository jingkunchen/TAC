# Semi-supervised Unpaired Medical Image Segmentation Through Task-affinity Consistency

This repository contains the code implementation for the paper titled "Semi-supervised Unpaired Medical Image Segmentation Through Task-affinity Consistency." In this work, we propose a novel approach for medical image segmentation using semi-supervised learning and task-affinity consistency.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

Medical image segmentation plays a critical role in various medical applications. This repository presents an approach that leverages semi-supervised learning and task-affinity consistency to improve the accuracy of medical image segmentation. The code provided here is based on the paper https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9915650.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- Pytorch version >=0.4.1.
- Other required packages (list them here)

### Installation

1. Clone this repository:

git clone https://github.com/jingkunchen/TAC.git

2. Install the required dependencies:

pip install -r requirements.txt

## Usage

To use this code, follow these steps:

1. Download your dataset, preprocess it as needed and put the data in data/2018LA_Seg_Training Set.

2. Run the training script:

python train_class_attention.py

3. Perform inference on your test data.

**Note**: The release version of the code in this repository has been optimized to remove unnecessary debugging and non-essential log information. Please feel free to modify it as needed.

## Training

The `train_class_attention.py` script serves as the main entry point for training your model. You can customize the training process by modifying the `config.yaml` file. This file contains all the hyperparameters and configuration settings.

## Inference

To perform inference on new data, use the test script. Ensure that you have trained the model and specified the appropriate checkpoint in the configuration file.

## Citation

If you find this code or our work helpful, please consider citing our paper:

@article{chen2023semi,
  title={Semi-supervised unpaired medical image segmentation through task-affinity consistency},
  author={Chen, Jingkun and Zhang, Jianguo and Debattista, Kurt and Han, Jungong},
  journal={IEEE Transactions on Medical Imaging},
  volume={42},
  number={3},
  pages={594--605},
  year={2023},
  publisher={IEEE}
}

## Acknowledgments

This code is adapted from UA-MT, SASSNet, SegWithDistMap, DTC.