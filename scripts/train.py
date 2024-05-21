import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import time
import copy
import face_recognition

# Define device to use GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

# Custom Dataset class for handling the Emotion data
class EmotionDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data = data_frame
        self.transform = transform

        # Ensure the DataFrame has at least two columns: labels and pixel data
        if self.data.shape[1] < 2:
            raise ValueError("DataFrame should have at least two columns: one for labels and one for pixel data.")

        # Filter out images without detectable faces
        self.filtered_indices = self.filter_faces()

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
        for idx in range(len(self.data)):
            pixels = self.data.iloc[idx, 1]
            try:
                pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
                # Check if the face is detected in the image
                if len(face_recognition.face_locations(pixels)) > 0:
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
        return valid_indices

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # TensorBoard setup for logging training process
    writer = SummaryWriter(log_dir='../runs/emotion_classification')

    if use_cuda:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
    else:
        scaler = None

    # Lists to store loss and accuracy for plotting learning curves
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if use_cuda:
                        with autocast():
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        if use_cuda:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve += 1

        print()

        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch} epochs')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    writer.close()

    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig('../results/models/learning_curves.png')
    plt.show()

    return model

# Read the train images with emotion labels from ../data/train.csv
data = pd.read_csv('../data/train.csv')

# Split the data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)

# Transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert images to grayscale with 3 channels
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Create the datasets
train_dataset = EmotionDataset(data_frame=train_data, transform=transform)
val_dataset = EmotionDataset(data_frame=val_data, transform=transform)

# Create the dataloaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=64, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=64, shuffle=True)
}

# Dataset sizes for computing loss and accuracy
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# Create the model (ResNet18) and modify the final layer to match the number of emotion classes
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # Assuming 7 emotion classes
model = model.to(device)

# Create the loss function (CrossEntropyLoss for classification)
criterion = nn.CrossEntropyLoss()

# Create the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Create the learning rate scheduler to adjust learning rate over epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=15)

# Save the trained model to disk
torch.save(model.state_dict(), '../results/models/emotion_model.pth')

# Test the model using test data
model.eval()

# Read the test images with emotion labels from ../data/test.csv
test_data = pd.read_csv('../data/test.csv')

# Create the test dataset
test_dataset = EmotionDataset(data_frame=test_data, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Evaluate the model on test data
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
print(f'Test Accuracy: {accuracy:.4f}')

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

# Check if accuracy meets the requirement
if accuracy >= 0.70:
    print("The model meets the accuracy requirement.")
else:
    print("The model does not meet the accuracy requirement.")
