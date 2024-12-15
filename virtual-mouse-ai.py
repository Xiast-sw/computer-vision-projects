import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
import numpy as np
import math
import time

# Mediapipe el izleme
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Fare kontrolcüsü
mouse = Controller()

# Ekran boyutları (2300x1200)
screen_width, screen_height = 2300, 1200

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Çerçeve boyutları
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# Tıklama için mesafe eşiği
click_threshold = 50  # Başparmak ve işaret parmağı arasındaki mesafe eşik değeri
double_click_threshold = 0.3  # Çift tıklama için zaman eşiği (saniye)

# Fare tıklama durumu
last_click_time = 0
last_double_click_time = 0
clicking = False
double_clicking = False

# İmleç hareketi için minimum mesafe (titreşimi engellemek için)
min_move_distance = 20  # Hareketin yapılabilmesi için minimum mesafe

# Hareket geçmişi
last_position = (0, 0)


def calculate_distance(point1, point2):
    """İki nokta arasındaki mesafeyi hesaplayan fonksiyon"""
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Çerçeveyi yansıtma ve BGR'den RGB'ye dönüştürme
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El tespiti
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # İşaret parmağı ve orta parmak noktaları
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Parmakların kamera koordinatları
            x1, y1 = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
            x2, y2 = int(middle_finger_tip.x * frame_width), int(middle_finger_tip.y * frame_height)

            # Kamera koordinatlarını ekran koordinatlarına ölçeklendirme
            screen_x = int(np.interp(x1, [0, frame_width], [0, screen_width]))
            screen_y = int(np.interp(y1, [0, frame_height], [0, screen_height]))

            # Hareketi kontrol etme: Ekrandaki önceki pozisyonla yeni pozisyon arasındaki fark
            if abs(screen_x - last_position[0]) > min_move_distance or abs(
                    screen_y - last_position[1]) > min_move_distance:
                mouse.position = (screen_x, screen_y)  # Fareyi hareket ettir
                last_position = (screen_x, screen_y)  # Yeni pozisyonu kaydet

            # İşaret parmağı ve orta parmak arasındaki mesafeyi hesapla
            distance = calculate_distance(index_finger_tip, middle_finger_tip)

            # Mesafeyi ekran üzerinde yazdırarak izleme
            cv2.putText(frame, f'Distance: {int(distance)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Eğer mesafe eşik değerinin altına düşerse tıklama işlemi yap
            if distance < click_threshold and (time.time() - last_click_time) > 1:  # 1 saniye bekle
                mouse.click(Button.left)  # Button'ı burada kullanıyoruz
                last_click_time = time.time()  # Tıklama zamanı güncelleniyor
                clicking = True

            # Çift tıklama kontrolü
            if distance < click_threshold and (time.time() - last_double_click_time) < double_click_threshold:
                mouse.click(Button.left)
                last_double_click_time = 0  # Çift tıklama yapıldı, zaman sıfırlanır
                double_clicking = True
            elif distance >= click_threshold:
                clicking = False
                double_clicking = False

            # Parmakları çizme (işaret parmağı ve orta parmak)
            cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)

            # El çizimi
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Çerçeveyi gösterme
    cv2.imshow("Virtual Mouse", frame)

    # Çıkış için 'q' tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
