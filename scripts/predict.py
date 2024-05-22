import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from PIL import Image
import face_recognition

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the EmotionDataset class
class EmotionDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data = data_frame
        self.transform = transform

        # Ensure the DataFrame has at least two columns: labels and pixel data
        if self.data.shape[1] < 2:
            raise ValueError("DataFrame should have at least two columns: one for labels and one for pixel data.")

        # Print the column names for debugging
        print(f"DataFrame columns: {self.data.columns}")

        # Filter out images that do not contain faces
        self.filtered_indices = self.filter_faces()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        pixels = self.data.iloc[actual_idx]['pixels']
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        image = Image.fromarray(pixels)

        label = int(self.data.iloc[actual_idx]['emotion'])
        if self.transform:
            image = self.transform(image)
        return image, label

    def filter_faces(self):
        valid_indices = []
        for idx in range(len(self.data)):
            pixels = self.data.iloc[idx]['pixels']
            pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
            if len(face_recognition.face_locations(pixels)) > 0:
                valid_indices.append(idx)
        return valid_indices

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels to match pretrained model input
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the test data
test_data = pd.read_csv('../data/test_with_emotions.csv')
# Drop the index column if it exists
test_data = test_data.drop(test_data.columns[0], axis=1, errors='ignore')

# Check the structure of the DataFrame
print(f"DataFrame head:\n{test_data.head()}")

# Ensure the DataFrame has the expected columns
expected_columns = ["emotion", "pixels"]
if not all(column in test_data.columns for column in expected_columns):
    raise ValueError(f"DataFrame must contain the following columns: {expected_columns}")

# Create the test dataset and dataloader
test_dataset = EmotionDataset(data_frame=test_data, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle for testing

# Load the model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
model.load_state_dict(torch.load('../results/models/emotion_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Predict on the test set
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

# Calculate the accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
