import cv2
import mediapipe as mp

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  # Capture frame

    if not ret:
        break  # Exit if error occurs

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:  # If hands detected
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Hand Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
