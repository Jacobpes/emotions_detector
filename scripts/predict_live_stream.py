import cv2
import face_recognition
import time
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Load the emotion prediction model
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)  # Assuming 7 emotion classes

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a function to load the model
def load_model(path):
    model = EmotionModel()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the pre-trained emotion prediction model
model_path = '../results/models/emotion_model.pth'
emotion_model = load_model(model_path)

# Define the emotion classes
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create output directory if it doesn't exist
output_dir = '../results/preprocessing_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the default webcam
cap = cv2.VideoCapture(1) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Reading video stream ...")
print("Press 'q' to quit the livestream.")

# Define the codec and create VideoWriter object for the video
video_path = os.path.join(output_dir, 'input_video.mp4')
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

# Initialize counter for images
image_counter = 0

# Record start time
start_time = time.time()

# Define a transformation to preprocess the face images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Write the frame into the video file without markings
    out.write(frame)

    # Convert the frame from BGR (OpenCV's format) to RGB (face_recognition's format)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Save face images with rectangles
    for face_location in face_locations:
        if image_counter < 20:
            top, right, bottom, left = face_location
            face = frame[top:bottom, left:right]
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            image_path = os.path.join(output_dir, f'image{image_counter}.png')
            cv2.imwrite(image_path, face)
            image_counter += 1

            # Draw rectangle around the face on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # Preprocess the face image and predict the emotion
            face_tensor = transform(face).unsqueeze(0)
            with torch.no_grad():
                emotion_output = emotion_model(face_tensor)
                emotion_prediction = torch.argmax(emotion_output, dim=1)
                emotion_label = emotion_classes[emotion_prediction.item()]
                emotion_prob = torch.softmax(emotion_output, dim=1)[0][emotion_prediction.item()].item() * 100

            # Put the emotion label and probability on the frame
            cv2.putText(frame, f"{emotion_label} , {emotion_prob:.0f}%", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Print the emotion label and probability
            current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"Preprocessing ...\n {current_time} : {emotion_label} , {emotion_prob:.0f}%")

    # Display the frame with detected faces and emotion labels
    cv2.imshow('Face Detection Livestream', frame)

    # Exit the livestream when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Stop after 20 seconds
    if time.time() - start_time >= 20:
        break

# Release the webcam and video writer
cap.release()
out.release()
cv2.destroyAllWindows()
