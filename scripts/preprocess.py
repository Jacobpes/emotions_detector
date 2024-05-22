import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import face_recognition

# Custom Dataset class for handling the Emotion data
class EmotionDataset(Dataset):
    def __init__(self, data_frame, transform=None, tolerance=0.6):
        print(f"Total images before filtering: {len(data_frame)}")
        self.data = data_frame
        self.transform = transform
        self.tolerance = tolerance
        self.filtered_indices, self.hog_filtered_count, self.cnn_filtered_count = self.filter_faces()
        print(f"Filtered {self.hog_filtered_count} images using HOG face detection")
        print(f"Filtered {self.cnn_filtered_count} images using CNN face detection")
        print(f"Total images after filtering: {len(self.filtered_indices)}")
        print(f"Total images filtered out: {len(self.data) - len(self.filtered_indices)}")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        pixels = self.data.iloc[actual_idx, 1]
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        image = Image.fromarray(pixels)
        label = int(self.data.iloc[actual_idx, 0])
        if self.transform:
            image = self.transform(image)
        return image, label

    def filter_faces(self):
        valid_indices = []
        hog_filtered_count = 0
        cnn_filtered_count = 0
        for idx in range(len(self.data)):
            pixels = self.data.iloc[idx, 1]
            try:
                pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
                face_locations_hog = face_recognition.face_locations(pixels, model="hog")
                face_locations_cnn = face_recognition.face_locations(pixels, model="cnn")
                if len(face_locations_hog) == 0:
                    hog_filtered_count += 1
                if len(face_locations_cnn) == 0:
                    cnn_filtered_count += 1
                if len(face_locations_hog) > 0 or len(face_locations_cnn) > 0:
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
        return valid_indices, hog_filtered_count, cnn_filtered_count

# Data preprocessing
def preprocess_data(csv_path, batch_size=64):
    data = pd.read_csv(csv_path)
    train_data, val_data = train_test_split(data, test_size=0.2)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = EmotionDataset(data_frame=train_data, transform=transform)
    val_dataset = EmotionDataset(data_frame=val_data, transform=transform)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    return dataloaders, dataset_sizes
