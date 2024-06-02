import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten  # type: ignore
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.optimizers import Nadam# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau# type: ignore
from tensorflow.keras.utils import to_categorical# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore
from tensorflow.keras.callbacks import TensorBoard# type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set GPU if available
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

# Load the training data from the CSV file and process it for training the model
def process_df(df):
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=float, sep=' '))
    X, y = [], []
    for _, row in df.iterrows():
        image = row['pixels'].reshape(48, 48, 1)
        X.append(image)
        y.append(row['emotion'])
    X = np.array(X) / 255.0
    y = to_categorical(np.array(y), num_classes=7)
    return X, y

def load_data(train_filepath):
    if not os.path.exists(train_filepath):
        print(f"Error: The file does not exist.")
        return None, None, None, None, None, None
    df_train = pd.read_csv(train_filepath)
    X_train, y_train = process_df(df_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=40)
    return X_train, X_val, y_train, y_val

# Sequential model definition
def create_model():
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(512, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_1'),
        BatchNormalization(name='batchnorm_1'),
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_2'),
        BatchNormalization(name='batchnorm_2'),
        MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'),
        Dropout(0.22, name='dropout_1'),
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_3'),
        BatchNormalization(name='batchnorm_3'),
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_4'),
        BatchNormalization(name='batchnorm_4'),
        MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'),
        Dropout(0.22, name='dropout_2'),
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_5'),
        BatchNormalization(name='batchnorm_5'),
        Conv2D(512, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_6'),
        BatchNormalization(name='batchnorm_6'),
        MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'),
        Dropout(0.22, name='dropout_3'),
        Flatten(name='flatten'),
        Dense(256, activation='elu', kernel_initializer='he_normal', name='dense_1'),
        BatchNormalization(name='batchnorm_7'),
        Dropout(0.22, name='dropout_4'),
        Dense(7, activation='softmax', name='out_layer')
    ])
    model.compile(optimizer=Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'), 
    loss='categorical_crossentropy', metrics=['accuracy']) 
    model.summary()
    return model

# Fit the model with the training data and validate with the validation data.
def fit_model(model, X_train, y_train, X_val, y_val):
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest'
    )
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.84, patience=7, min_lr=0.0003, verbose=1),
        ModelCheckpoint('drive/MyDrive/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True), 
        EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    ]
    history = model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=120,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=2)
    return history

def main():
    configure_gpu()
    train_filepath = 'drive/MyDrive/train.csv'
    X_train, X_val, y_train, y_val = load_data(train_filepath)

    if X_train is not None:
        model = create_model()
        history = fit_model(model, X_train, y_train, X_val, y_val)
        model.save('drive/MyDrive/best_model.h5')
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('learning_curves.png')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('loss_curves.png')
        plt.show()

if __name__ == "__main__":
    main()
