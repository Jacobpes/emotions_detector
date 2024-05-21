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

2. Predicting Emotions on the Test Set
To predict emotions on the test set and calculate accuracy, use the predict.py script:

sh
Kopiera kod
python ./scripts/predict.py
Expected output:

sh
Kopiera kod
Accuracy on test set: 72%
3. Predicting Emotions from a Live Video Stream
To predict emotions from a live video stream, use the predict_live_stream.py script:

Model Explanation
Emotion Classification Model
The emotion classification model is based on a Convolutional Neural Network (CNN) architecture using ResNet-18. The model is designed to classify facial emotions into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Model Architecture
Base Model: ResNet-18, a widely used deep CNN architecture known for its simplicity and effectiveness.
Modifications:
The final fully connected layer of ResNet-18 is replaced with a new fully connected layer that has seven output nodes, each corresponding to one of the seven emotion categories.
This allows the model to output probabilities for each emotion class.
Training Process
Data Loading:

The training data is loaded from ../data/train.csv.
Images are preprocessed, including converting to grayscale, resizing to 224x224 pixels, and normalizing pixel values.
Model Training:

The model is trained using a cross-entropy loss function.
The optimizer used is stochastic gradient descent (SGD) with a learning rate scheduler to adjust the learning rate over time.
Early Stopping:

To prevent overfitting, early stopping is implemented based on validation accuracy. Training stops if the model does not improve for a set number of epochs.
Monitoring:

TensorBoard is used to monitor the training process, allowing visualization of training and validation loss and accuracy over epochs.
Preprocessing
The preprocessing steps include:

Converting images to grayscale: To simplify the input data.
Resizing images to 224x224 pixels: To match the input size required by ResNet-18.
Normalizing pixel values: To standardize the input data, making training more efficient.
Prediction
For predicting emotions on the test set and live video streams:

The trained model is loaded.
Images are preprocessed in the same manner as the training data.
The model outputs the predicted emotion and the associated probability for each input image.
