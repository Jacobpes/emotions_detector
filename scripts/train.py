import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import face_recognition
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import cv2  # OpenCV
import dlib
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
import face_recognition

def load_data_with_face_filter(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
    
    X = []
    y = []

    for index, row in df.iterrows():
        image = row['pixels'].reshape(48, 48).astype('uint8')
        
        # Start with the most accurate and resource-intensive model
        face_locations_cnn = face_recognition.face_locations(image, model="cnn")

        # If CNN model finds a face, use this result
        if face_locations_cnn:
            X.append(row['pixels'] / 255.0)  # Normalize pixel values
            y.append(row['emotion'])
        else:
            # If no face detected by CNN, use HOG
            face_locations_hog = face_recognition.face_locations(image, model="hog")
            if face_locations_hog:
                X.append(row['pixels'] / 255.0)
                y.append(row['emotion'])

    X = np.array(X).reshape(-1, 48, 48, 1)  # Reshape for CNN, 1 channel for grayscale
    y = np.array(y)

    # print how many pics were filtered out
    print(f"Filtered out {len(df) - len(X)} images without faces")

    # Handling imbalance in the dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(-1, 48 * 48), y)
    X_resampled = X_resampled.reshape(-1, 48, 48, 1)
    y_resampled = to_categorical(y_resampled)

    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


def build_model():
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.50),
        Dense(7, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, X_val, y_train, y_val):
    # Compute class weights for unbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weight_dict = dict(enumerate(class_weights))

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=70,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2
    )

    return history

def main():
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_data_with_face_filter('../data/train.csv')

    # Build model
    model = build_model()

    # Train the model
    history = train_model(model, X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()