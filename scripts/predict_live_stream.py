import numpy as np
import cv2
import face_recognition
import time
import os
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained emotion prediction model from the specified path with the custom objects
model_path = '../results/models/best_model.h5'
emotion_model = load_model(model_path)

# Define the emotion classes that the model can recognize
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create an output directory for videos or images
output_dir = '../results/preprocessing_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the webcam. For some reason the builtin cam on Grit:lab mac is 1
cap = cv2.VideoCapture(1)

# Check if the webcam starts successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Reading video stream ...")
print("Press 'q' to quit the livestream.")

# Define the codec and settings for video writer object to save output video
video_path = os.path.join(output_dir, 'input_video.mp4')
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 300, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture image.")
        break

    out.write(frame)  # Write the frame to the output video file

    rgb_frame = frame[:, :, ::-1]  # Convert the image from BGR to RGB format
    face_locations = face_recognition.face_locations(rgb_frame)  # Detect faces in the frame

    for face_location in face_locations:
        top, right, bottom, left = face_location  # Unpack the coordinates of the face
        face = frame[top:bottom, left:right]  # Extract the face from the frame
        face = cv2.resize(face, (48, 48))  # Resize the face to the size expected by the model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert the face to grayscale
        face = np.expand_dims(face, axis=-1)  # Add a channel dimension
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize pixel values and add a batch dimension

        emotion_prediction = emotion_model.predict(face)  # Predict the emotion of the detected face
        emotion_label_index = np.argmax(emotion_prediction)  # Find the index of the highest prediction
        emotion_label = emotion_classes[emotion_label_index]  # Map the index to the corresponding emotion
        emotion_prob = np.max(emotion_prediction) * 100  # Calculate the probability of the predicted emotion

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # Draw a rectangle around the face
        cv2.putText(frame, f"{emotion_label}, {emotion_prob:.0f}%", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Display the emotion and probability

        current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))  # Format the current time
        print(f"Preprocessing ...\n {current_time} : {emotion_label} , {emotion_prob:.0f}%")  # Print the emotion label and probability with timestamp

    cv2.imshow('Face Detection Livestream', frame)  # Show the frame with the detected faces and emotions

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

    if time.time() - start_time >= 300:  # Stop after 5 minutes for testing
        break

# Release the webcam and video writer resources
cap.release()
out.release()
cv2.destroyAllWindows()
