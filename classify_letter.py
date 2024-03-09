import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_path = './model.p'
with open(model_path, 'rb') as file:
    model_dict = pickle.load(file)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,  # Use dynamic mode
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Video capture
cap = cv2.VideoCapture(2)

while True:
    success, frame = cap.read()
    if not success:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Extract features
            landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            min_x, min_y = landmarks.min(axis=0)
            landmarks_normalized = landmarks - [min_x, min_y]

            # Prepare the data for prediction
            data_aux = landmarks_normalized.flatten()

            # Make a prediction
            prediction = model.predict([data_aux])
            predicted_character = labels_dict[int(prediction[0])]

            # Calculate bounding box coordinates
            x1, y1 = (landmarks.min(axis=0) * [W, H]).astype(int) - 10
            x2, y2 = (landmarks.max(axis=0) * [W, H]).astype(int) + 10

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
