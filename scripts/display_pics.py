import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot

def load_data_with_face_filter(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    X = []
    y = []

    # Filter out images without faces
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

    # Reshape for CNN, 1 channel for grayscale
    X = np.array(X).reshape(-1, 48, 48, 1)
    # One-hot encoding of labels for multi-class classification
    y = to_categorical(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load the training data
# train_data = load_data_with_face_filter('../data/train.csv')
train_data = pd.read_csv('../data/train.csv')

emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# print each emotion and the number of examples
for i in range(7):
    print(f'[{i}] {emotion_classes[i]}: {len(train_data[train_data["emotion"] == i])}')

for i in range(7):
    # # Display a sample of 49 images from the training data with emotion i
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(emotion_classes[i], fontsize=20)
    for j in range(49):
        pixels = train_data[train_data['emotion'] == i].iloc[j]['pixels']
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        ax = fig.add_subplot(7, 7, j + 1)
        ax.imshow(pixels, cmap='gray')
        ax.axis('off')
    plt.show()

# Display one at a time of specified emotion
emotion = 3  # Happy
for i in range(len(train_data)):
    # if emotion is disgust:
    if train_data.iloc[i]['emotion'] == 3:
        pixels = train_data.iloc[i]['pixels']
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        plt.imshow(pixels, cmap='gray')
        plt.title(emotion_classes[train_data.iloc[i]['emotion']])
        plt.show()