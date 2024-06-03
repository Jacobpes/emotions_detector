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

```sh
python ./scripts/predict.py
```

3. Predicting Emotions from a Live Video Stream
To predict emotions from a live video stream, use the predict_live_stream.py script:

```sh
python ./scripts/predict_live_stream.py
```

# Architecture for Emotion Recognition Model

I first tried with vgg but failed to score enough. Then I tested the Sequential model and found out that it works better. I got inspiration from [This project on Github](https://www.kaggle.com/code/farneetsingh24/ck-facial-emotion-recognition-96-46-accuracy)

High Depth Convolutional Layers: The use of multiple convolutional layers, particularly the alternating high-depth layers (512 and 256 filters in the initial layers), allows the model to capture complex and subtle features in the facial expressions. High-depth filters increase the model’s ability to learn detailed nuances in the data which is critical for accurately categorizing emotions.

# When i fine tuned I pinpointed these parameters:

🧠ELU > RELU
ELU Activation: ELU (Exponential Linear Unit) activation functions are used instead of more common ReLU to help the model learn and converge faster. ELU helps in reducing the vanishing gradient problem, which is beneficial when training deeper networks like this

🧠DROPOUT PINPOINTED TO 22
Dropout and MaxPooling: Strategically placed dropout layers at 23% rate help in preventing overfitting by randomly dropping units during the training process, which promotes the development of a more generalized model. MaxPooling is used to reduce the spatial dimensions of the output volumes, which not only helps in reducing the computational load but also in extracting dominant features which are invariant to small changes in the input space.

🧠ROTATION PINPOINTED TO 17 DEGREES
Data Augmentation: The image data generator manipulates the training images through rotations, shifts, and flips, which helps in creating a robust model that is less sensitive to variations in input data, reflecting real-world scenarios where facial expressions can vary significantly in appearance.

🧠NADAM > ADAM
Optimizer and Learning Rate Adjustments: The Nadam optimizer is chosen for its effectiveness in handling sparse gradients and preventing premature convergence. The learning rate adjustments through ReduceLROnPlateau callback ensure that the model fine-tunes its parameters more delicately as it approaches optimal performance, enhancing the accuracy of the model without overshooting during training.

🧠BATCH SIZE PINPOINTED TO 64
🧠2X 128 LAYERS IN THE MIDDLE WORKS LIKE AN ATTENTION MECHANISM
🧠stratify=y_train -> SMALLER TEST SET 

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_1 (Conv2D)           (None, 48, 48, 512)       5120      
                                                                 
 batchnorm_1 (BatchNormaliz  (None, 48, 48, 512)       2048      
 ation)                                                          
                                                                 
 conv2d_2 (Conv2D)           (None, 48, 48, 256)       1179904   
                                                                 
 batchnorm_2 (BatchNormaliz  (None, 48, 48, 256)       1024      
 ation)                                                          
                                                                 
 maxpool2d_1 (MaxPooling2D)  (None, 24, 24, 256)       0         
                                                                 
 dropout_1 (Dropout)         (None, 24, 24, 256)       0         
                                                                 
 conv2d_3 (Conv2D)           (None, 24, 24, 128)       295040    
                                                                 
 batchnorm_3 (BatchNormaliz  (None, 24, 24, 128)       512       
 ation)                                                          
                                                                 
 conv2d_4 (Conv2D)           (None, 24, 24, 128)       147584    
                                                                 
 batchnorm_4 (BatchNormaliz  (None, 24, 24, 128)       512       
 ation)                                                          
                                                                 
 maxpool2d_2 (MaxPooling2D)  (None, 12, 12, 128)       0         
                                                                 
 dropout_2 (Dropout)         (None, 12, 12, 128)       0         
                                                                 
 conv2d_5 (Conv2D)           (None, 12, 12, 256)       295168    
                                                                 
 batchnorm_5 (BatchNormaliz  (None, 12, 12, 256)       1024      
 ation)                                                          
                                                                 
 conv2d_6 (Conv2D)           (None, 12, 12, 512)       1180160   
                                                                 
 batchnorm_6 (BatchNormaliz  (None, 12, 12, 512)       2048      
 ation)                                                          
                                                                 
 maxpool2d_3 (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                 
 dropout_3 (Dropout)         (None, 6, 6, 512)         0         
                                                                 
 flatten (Flatten)           (None, 18432)             0         
                                                                 
 dense_1 (Dense)             (None, 256)               4718848   
                                                                 
 batchnorm_7 (BatchNormaliz  (None, 256)               1024      
 ation)                                                          
                                                                 
 dropout_4 (Dropout)         (None, 256)               0         
                                                                 
 out_layer (Dense)           (None, 7)                 1799      
                                                                 
=================================================================
Total params: 7831815 (29.88 MB)
Trainable params: 7827719 (29.86 MB)
Non-trainable params: 4096 (16.00 KB)

Jacob Pesämaa, 2024
