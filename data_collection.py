import os
import cv2
import time

# Configuration
DATA_DIR = './data'
number_of_classes = 26
dataset_size = 100
video_source = 0  # Change this according to your video capture device

# Ensure the base data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize video capture
cap = cv2.VideoCapture(video_source)


# Function to capture images for a single class
def capture_images_for_class(class_id, num_images):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_id}')

    # Ready message
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Give user time to prepare
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, str(countdown), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 4)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

    # Capture images
    for img_num in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('frame', frame)
        img_path = os.path.join(class_dir, f'{img_num}.jpg')
        cv2.imwrite(img_path, frame)
        cv2.waitKey(25)  # You might adjust this if you need more time between captures


# Main loop for capturing images for each class
for class_id in range(number_of_classes):
    capture_images_for_class(class_id, dataset_size)

# Release resources
cap.release()
cv2.destroyAllWindows()
