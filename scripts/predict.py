# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply, Add, Activation, Conv2D
import face_recognition

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


def load_test_data(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    X = []
    y = []
    images = []

    for index, row in df.iterrows():
        image = row['pixels'].reshape(48, 48).astype('uint8')
        face_locations = face_recognition.face_locations(image, model="hog")
        if face_locations:
            X.append(row['pixels'] / 255.0)  # Normalize pixel values
            y.append(row['emotion'])
            images.append(image)  # Save original image for any potential review

    X = np.array(X).reshape(-1, 48, 48, 1)  # Reshape for CNN, 1 channel for grayscale
    y = np.array(y)
    return X, y, images

def predict_emotions(model, X):
    predictions = model.predict(X)
    return predictions

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def main():
    # Load test data
    test_filepath = '../data/test_with_emotions.csv'
    X_test, y_test, test_images = load_test_data(test_filepath)
    
    # Load the model
    model_path = '../results/models/best_model.keras'
    model = load_model(model_path, custom_objects={'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention})

    # Make predictions
    predictions = predict_emotions(model, X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, predicted_classes)
    print(f'Accuracy of predictions: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
