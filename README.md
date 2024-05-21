# Emotions Detection with Deep Learning

## Introduction

This project focuses on detecting facial emotions using Convolutional Neural Networks (CNNs). The goal is to implement a system that detects the emotion on a face from a video stream. The primary tasks are:

1. Emotion classification
2. Face tracking

## How to Run the Code

### 1. Training the Model

To train the emotion classification model, use the `train.py` script:

```sh
python ./scripts/train.py

```

To predict on the live stream use the `predict_live_stream.py`

```sh
python ./scripts/predict_live_stream.py
```