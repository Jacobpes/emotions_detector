# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import face_recognition
import cv2
import mediapipe as mp
import warnings
import pickle

def extract_face_mesh(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    result = face_mesh.process(image_rgb)
    if not result.multi_face_landmarks:
        return np.zeros(468 * 3)  # 468 landmarks, 3 coordinates (x, y, z)

    landmarks = result.multi_face_landmarks[0].landmark
    mesh_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return mesh_data

def load_test_data(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    X = []
    y = []
    images = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    for _, row in df.iterrows():
        image = row['pixels'].reshape(48, 48).astype('uint8')
        face_locations = face_recognition.face_locations(image, model="hog")
        if face_locations:
            mesh_data = extract_face_mesh(image, face_mesh)
            combined_data = np.concatenate((image.flatten(), mesh_data))
            X.append(combined_data / 255.0)
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
    model_path = '../results/sexiest_model_alive.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    predictions = predict_emotions(model, X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, predicted_classes)
    print(f'Accuracy of predictions: {accuracy:.2f}%')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()