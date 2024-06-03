import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the training data
X_train = pd.read_csv('../data/train.csv')
X_train['pixels'] = X_train['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# print each emotion and the number of examples
for i in range(7):
    print(f'[{i}] {emotion_classes[i]}: {len(X_train[X_train["emotion"] == i])}')

# Display one at a time of specified emotion
emotion = 3  # Happy
for i in range(len(X_train)):
    # if emotion is disgust:
    if X_train.iloc[i]['emotion'] == 3:
        pixels = X_train.iloc[i]['pixels']
        pixels = np.array(pixels, dtype='uint8').reshape(48, 48)
        plt.imshow(pixels, cmap='gray')
        plt.title(emotion_classes[X_train.iloc[i]['emotion']])
        plt.show()