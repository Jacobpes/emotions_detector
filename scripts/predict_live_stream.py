import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import face_recognition
import time
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Layer, Multiply, Add, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = Dense(input_shape[-1] // self.ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_layer_two = Dense(input_shape[-1],
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        super(ChannelAttention, self).build(input_shape)

    def call(self, input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        avg_pool = Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_tensor)
        max_pool = Reshape((1, 1, max_pool.shape[1]))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        scale = Activation('sigmoid')(Add()([avg_pool, max_pool]))
        return Multiply()([input_tensor, scale])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv2d = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, input_tensor):
        avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return Multiply()([input_tensor, self.conv2d(concat)])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
    def build(self, input_shape):
        self.conv2d = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super(SpatialAttention, self).build(input_shape)  # Mark the layer as built


# Map the custom objects
custom_objects = {
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention
}

# Load the pre-trained emotion prediction model from the specified path with the custom objects
model_path = '../results/models/best_model.keras'
emotion_model = load_model(model_path, custom_objects=custom_objects)

# Define the emotion classes that the model can recognize
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create an output directory for any results like video or images
output_dir = '../results/preprocessing_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the webcam; ensure the correct index is used (commonly 0 for the default webcam)
cap = cv2.VideoCapture(1)

# Check if the webcam starts successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Reading video stream ...")
print("Press 'q' to quit the livestream.")

# Define the codec and settings for video writer object to save output video
video_path = os.path.join(output_dir, 'input_video.mp4')
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time()  # Record the starting time for timestamps

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture image.")
        break

    out.write(frame)  # Write the frame to the output video file

    rgb_frame = frame[:, :, ::-1]  # Convert the image from BGR to RGB format
    face_locations = face_recognition.face_locations(rgb_frame)  # Detect faces in the frame

    for face_location in face_locations:
        top, right, bottom, left = face_location  # Unpack the coordinates of the face
        face = frame[top:bottom, left:right]  # Extract the face from the frame
        face = cv2.resize(face, (48, 48))  # Resize the face to the size expected by the model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert the face to grayscale
        face = np.expand_dims(face, axis=-1)  # Add a channel dimension
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize pixel values and add a batch dimension

        emotion_prediction = emotion_model.predict(face)  # Predict the emotion of the detected face
        emotion_label_index = np.argmax(emotion_prediction)  # Find the index of the highest prediction
        emotion_label = emotion_classes[emotion_label_index]  # Map the index to the corresponding emotion
        emotion_prob = np.max(emotion_prediction) * 100  # Calculate the probability of the predicted emotion

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # Draw a rectangle around the face
        cv2.putText(frame, f"{emotion_label}, {emotion_prob:.0f}%", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Display the emotion and probability

        current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))  # Format the current time
        print(f"Preprocessing ...\n {current_time} : {emotion_label} , {emotion_prob:.0f}%")  # Print the emotion label and probability with timestamp
        # make it less laggy
        time.sleep(0.1)

    cv2.imshow('Face Detection Livestream', frame)  # Show the frame with the detected faces and emotions

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

    if time.time() - start_time >= 300:  # Stop after 5 minutes for testing
        break

# Release the webcam and video writer resources
cap.release()
out.release()
cv2.destroyAllWindows()
