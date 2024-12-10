import cv2
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# MediaPipe settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Audio control settings
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Distance range
MIN_DISTANCE = 0.02
MAX_DISTANCE = 0.2

# Start the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Scale coordinates based on frame size
            h, w, _ = frame.shape
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))

            # Calculate the distance
            distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            # Draw a line (green color)
            cv2.line(frame, thumb_coords, index_coords, (0, 255, 0), 3)

            # Convert distance to volume level
            if distance < MIN_DISTANCE:
                volume_value = 0.0
            elif distance > MAX_DISTANCE:
                volume_value = 1.0
            else:
                volume_value = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)

            # Adjust the volume level
            volume.SetMasterVolumeLevelScalar(volume_value, None)

    cv2.imshow("Hand Volume Control with Green Line", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
