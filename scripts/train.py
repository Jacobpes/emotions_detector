import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pickle

# Configure GPU settings
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("CUDA is available. Configured TensorFlow to use the GPU.")
        except RuntimeError as e:
            print(f"Failed to set memory growth: {e}")
    else:
        print("CUDA is not available. TensorFlow will use CPU.")

# Process DataFrame to extract features and labels
def process_df(df):
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=float, sep=' '))
    X, y = [], []
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    
    for _, row in df.iterrows():
        image = row['pixels'].reshape(48, 48, 1)
        mesh_data = extract_face_mesh(image, face_mesh)
        combined_data = np.concatenate((image.flatten(), mesh_data))
        X.append(combined_data)
        y.append(row['emotion'])
    
    X = np.array(X) / 255.0
    y = to_categorical(np.array(y), num_classes=7)
    return X, y

def extract_face_mesh(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    result = face_mesh.process(image_rgb)
    if not result.multi_face_landmarks:
        return np.zeros(468 * 3)  # 468 landmarks, 3 coordinates (x, y, z)
    
    landmarks = result.multi_face_landmarks[0].landmark
    mesh_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return mesh_data

# Load training data and split into training and validation sets
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} does not exist.")
        return None, None, None, None
    df = pd.read_csv(filepath)
    X, y = process_df(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=40)
    return X_train, X_val, y_train, y_val

# Create and compile the model
def create_model(input_shape):
    image_input = Input(shape=(48 * 48 + 468 * 3,))
    x = Dense(1024, activation='elu', kernel_initializer='he_normal')(image_input)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='elu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='elu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=image_input, outputs=output)
    model.compile(optimizer=Nadam(learning_rate=0.001), 
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
                  metrics=['accuracy'])
    model.summary()
    return model

# Train the model and save history
def fit_model(model, X_train, y_train, X_val, y_val):
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest'
    )
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.84, patience=7, min_lr=0.0003, verbose=1),
        ModelCheckpoint('best_model.pkl', monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True), 
        EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    ]
    history = model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=120,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=2)
    return history

# Plot and save learning curves
def plot_learning_curves(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('learning_curves.png')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('loss_curves.png')
    plt.show()

def main():
    configure_gpu()
    train_filepath = 'drive/MyDrive/train.csv'
    X_train, X_val, y_train, y_val = load_data(train_filepath)

    if X_train is not None:
        input_shape = X_train.shape[1:]
        model = create_model(input_shape)
        history = fit_model(model, X_train, y_train, X_val, y_val)
        
        with open('../results/maybe_the_sexiest_model_alive.pkl', 'wb') as file:
            pickle.dump(model, file)

        plot_learning_curves(history)

if __name__ == "__main__":
    main()