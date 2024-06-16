import numpy as np
import cv2
import face_recognition
import time
import os
from tensorflow.keras.models import load_model  # type: ignore

# Load the pre-trained emotion detection model
model_path = '../results/models/sexiest_model_alive.pkl'
emotion_model = load_model(model_path)

# Emotion labels that the model can predict
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set up directory for saving outputs
output_dir = '../results/preprocessing_test'
os.makedirs(output_dir, exist_ok=True)

# Start the webcam (camera index may vary depending on the system)
camera_index = 1
cap = cv2.VideoCapture(camera_index)

# Check if the webcam initialized correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Starting video stream...")
print("Press 'q' to exit.")

# Configure video writer to save the output
video_filename = os.path.join(output_dir, 'input_video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

start_time = time.time()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Unable to capture frame.")
        break

    video_writer.write(frame)  # Save the frame to the video file

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    face_locations = face_recognition.face_locations(rgb_frame)  # Detect faces in the frame

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face = frame[top:bottom, left:right]  # Extract face region
        face = cv2.resize(face, (48, 48))  # Resize face to 48x48 pixels
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert face to grayscale
        face = face.astype('float32') / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict the emotion of the face
        emotion_prediction = emotion_model.predict(face)
        emotion_index = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[emotion_index]
        emotion_confidence = np.max(emotion_prediction) * 100

        # Draw rectangle around the face and label it with emotion and confidence
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_label}, {emotion_confidence:.0f}%", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"{elapsed_time} - Emotion: {emotion_label}, Confidence: {emotion_confidence:.0f}%")

    # Display the video frame with emotion annotations
    cv2.imshow('Emotion Detection Live Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

    if time.time() - start_time >= 300:  # Automatically stop after 5 minutes
        break

# Release resources and close windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()