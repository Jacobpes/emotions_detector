# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential  # type: ignore
import face_recognition
import warnings

def load_test_data(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    X = []
    y = []
    images = []

    for _, row in df.iterrows():
        image = row['pixels'].reshape(48, 48).astype('uint8')
        face_locations = face_recognition.face_locations(image, model="hog")
        if face_locations:
            X.append(image / 255.0)
            y.append(row['emotion'])
            images.append(image)

    X = np.array(X)
    y = np.array(y)
    return X, y

def predict_emotions(model, X):
    predictions = model.predict(X)
    return predictions

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def main():
    # Load test data
    test_filepath = '../data/test_with_emotions.csv'
    X_test, y_test = load_test_data(test_filepath)

    # Load and compile model
    model_path = '../results/models/best_model_epoch_88.h5'
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Make predictions
    predictions = predict_emotions(model, X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, predicted_classes)
    print(f'Accuracy of predictions: {accuracy:.2f}%')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
