import cv2
import mediapipe as mp

# MediaPipe modelini başlat
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# El izleyicisini başlat
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Çizim penceresini başlat
cap = cv2.VideoCapture(0)
drawing = False
last_point = None

# Çizilen çizgileri tutacak boş bir görüntü oluştur
drawing_image = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü BGR'den RGB'ye çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El izleme modelini uygula
    results = hands.process(frame_rgb)

    # Eğer bir el tespit edildiyse
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Başparmak ve işaret parmağı uçlarını al
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Elin büyüklüğüne göre piksel cinsinden konumları hesapla
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Başparmak ve işaret parmağı uçları birleştiğinde çizim başlat
            if abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:
                drawing = True
                if last_point:
                    # Çizilen çizgileri yeni bir görüntüde tutuyoruz
                    if drawing_image is None:
                        drawing_image = frame.copy()  # İlk çizim yapılırken boş bir çizim görüntüsü oluşturuluyor.
                    cv2.line(drawing_image, last_point, (thumb_x, thumb_y), (0, 255, 0), 5)
                last_point = (thumb_x, thumb_y)
            else:
                drawing = False
                last_point = None

            # Elin işaretini çiz
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Çizim yapılırken sadece çizim görüntüsünü ekrana yansıt
    if drawing_image is not None:
        frame = drawing_image

    # Sonuçları ekranda göster
    cv2.imshow("Drawing with Thumb and Index Finger", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
