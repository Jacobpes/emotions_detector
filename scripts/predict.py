import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import face_recognition

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
    model_path = 'best_model.keras'
    model = load_model(model_path)

    # Make predictions
    predictions = predict_emotions(model, X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, predicted_classes)
    print(f'Accuracy of predictions: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
