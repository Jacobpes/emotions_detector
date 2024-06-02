import pandas as pd
import numpy as np
import face_recognition

# Filters out the images that do not contain a face using the HOG and CNN models 
# and filters doubles.

def load_data_with_face_filter(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    X = []
    y = []
    original_indices = []

    for index, row in df.iterrows():
        image = row['pixels'].reshape(48, 48).astype('uint8')
        # Try to detect face with HOG and if not found, try CNN
        if not face_recognition.face_locations(image, model="hog"):
            if not face_recognition.face_locations(image, model="cnn"):
                original_indices.append(index)
                X.append(' '.join(row['pixels'].astype(str)))  # Convert pixels to space-separated string
                y.append(row['emotion'])

    return X, y, original_indices

print("Loading data with face filter...")
train_filepath = '../data/train.csv'
X_train, y_train, original_indices = load_data_with_face_filter(train_filepath)
print("Data loaded with face filter.", len(X_train), "samples found.")

# Create DataFrame with 3 columns
df = pd.DataFrame({
    'original_index': original_indices,  # Original index from 'train.csv
    'emotion': y_train,
    'pixels': X_train
})

df.to_csv('../data/train_without_faces_cnn_hog.csv', index=False)
print("Data saved to train_without_faces_cnn_hog.csv")

print("df describe", df.info(), df.head())
# Filter out the images that are doubles in the train.csv file
# read train.csv 
df = pd.read_csv('../data/train.csv')
lengthdf = len(df)
print("len df before dropping", len(df))
# check how many doubles there is in the dataframe
print("Number of doubles in the dataframe:", df.duplicated().sum())
df = df.drop_duplicates()
# write the filtered data to a new csv file data/train_filtered.csv
df.to_csv('../data/train_filtered.csv', index=False)
print("len df after dropping", len(df))