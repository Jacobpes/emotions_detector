import cv2
import face_recognition
import time
import os

# Create output directory if it doesn't exist
output_dir = '../results/preprocessing_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the default webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the livestream.")

# Define the codec and create VideoWriter object for the video
video_path = os.path.join(output_dir, 'input_video.mp4')
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

# Initialize counter for images
image_counter = 0

# Record start time
start_time = time.time()

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

    # Display the number of faces found
    print(f"Found {len(face_locations)} face(s) in the frame.")

    # Display the frame with detected faces
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
